from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


class CachePriorRouter:
    """Training-free cache-aware top-k router with top-j protection."""

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        top_j: int = 1,
        lam: float = 0.5,
        alpha: float = 0.99,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_j <= 0:
            raise ValueError("top_j must be positive")
        if top_j > top_k:
            raise ValueError("top_j must be <= top_k")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")

        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.top_j = min(int(top_j), self.top_k)
        self.lam = float(lam)
        self.alpha = float(alpha)

        self._delta_avg: Dict[int, float] = {}
        self._cache_prior_overrides = 0
        self._total_tokens = 0

    def route(
        self,
        logits: torch.Tensor,
        layer_id: int,
        cache_state: Iterable[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_f = logits.float()
        if logits_f.ndim == 1:
            logits_f = logits_f.unsqueeze(0)
        if logits_f.shape[-1] != self.num_experts:
            raise ValueError(
                f"Expected logits with last dimension {self.num_experts}, got {tuple(logits_f.shape)}"
            )

        batch_size = logits_f.shape[0]
        delta = logits_f.max(dim=-1).values - logits_f.min(dim=-1).values
        delta_mean = float(delta.mean().item())
        prev = self._delta_avg.get(layer_id, delta_mean)
        delta_avg = self.alpha * prev + (1.0 - self.alpha) * delta_mean
        self._delta_avg[layer_id] = delta_avg

        cache_mask = torch.zeros(self.num_experts, device=logits_f.device, dtype=logits_f.dtype)
        for expert_idx in cache_state:
            if 0 <= int(expert_idx) < self.num_experts:
                cache_mask[int(expert_idx)] = 1.0

        boosted = logits_f + (self.lam * delta_avg * cache_mask).unsqueeze(0)

        original_topk = torch.topk(logits_f, k=self.top_k, dim=-1).indices
        boosted_topk = torch.topk(boosted, k=self.top_k, dim=-1).indices
        protected_topj = torch.topk(logits_f, k=self.top_j, dim=-1).indices
        final_idx = boosted_topk.clone()

        for row in range(batch_size):
            row_final = final_idx[row].tolist()
            protected = protected_topj[row].tolist()
            for expert_idx in protected:
                if expert_idx in row_final:
                    continue
                selected_scores = boosted[row, final_idx[row]]
                removable_positions = [
                    pos for pos, selected_expert in enumerate(row_final)
                    if selected_expert not in protected
                ]
                if not removable_positions:
                    removable_positions = list(range(len(row_final)))
                remove_pos = min(removable_positions, key=lambda pos: float(selected_scores[pos].item()))
                row_final[remove_pos] = int(expert_idx)
                final_idx[row] = torch.tensor(row_final, device=final_idx.device, dtype=final_idx.dtype)

            selected_logits = logits_f[row, final_idx[row]]
            reorder = torch.argsort(selected_logits, descending=True)
            final_idx[row] = final_idx[row, reorder]

            if set(int(v) for v in final_idx[row].tolist()) != set(int(v) for v in original_topk[row].tolist()):
                self._cache_prior_overrides += 1

        probs = torch.softmax(logits_f, dim=-1)
        weights = torch.gather(probs, dim=-1, index=final_idx)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        self._total_tokens += batch_size
        return final_idx, weights

    @property
    def stats(self) -> Dict[str, object]:
        override_rate = (
            self._cache_prior_overrides / self._total_tokens
            if self._total_tokens
            else 0.0
        )
        return {
            "cache_prior_overrides": self._cache_prior_overrides,
            "cache_prior_tokens": self._total_tokens,
            "cache_prior_override_rate": override_rate,
            "cache_prior_delta_avg": dict(self._delta_avg),
            "cache_prior_lambda": self.lam,
            "cache_prior_top_j": self.top_j,
        }
