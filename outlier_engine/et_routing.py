from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import torch


class ETRouter:
    """Expert-threshold router with per-expert EMA probability thresholds."""

    def __init__(
        self,
        n_experts: int,
        top_k_fallback: int = 2,
        alpha: float = 0.99,
        min_experts: int = 1,
        max_experts: int = 4,
    ) -> None:
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if min_experts <= 0:
            raise ValueError("min_experts must be positive")
        if max_experts < min_experts:
            raise ValueError("max_experts must be >= min_experts")

        self.n_experts = int(n_experts)
        self.top_k_fallback = int(top_k_fallback)
        self.alpha = float(alpha)
        self.min_experts = int(min_experts)
        self.max_experts = min(int(max_experts), self.n_experts)

        self._thresholds = torch.zeros(self.n_experts, dtype=torch.float32)
        self._calibrated = False
        self._total_tokens_routed = 0
        self._total_experts_selected = 0
        self._expert_count_histogram: Counter[int] = Counter()

    @property
    def thresholds(self) -> torch.Tensor:
        return self._thresholds

    def calibrate(self, router_logits_batch: torch.Tensor) -> torch.Tensor:
        probs = self._to_probs(router_logits_batch)
        self._thresholds = probs.mean(dim=0).detach().to(dtype=torch.float32, device="cpu")
        self._calibrated = True
        return self._thresholds

    def route(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self._to_probs(router_logits)
        if not self._calibrated:
            self.calibrate(router_logits)

        thresholds = self._thresholds.to(device=probs.device, dtype=probs.dtype)
        selected_mask = probs > thresholds.unsqueeze(0)
        batch_size = probs.shape[0]
        fallback_k = min(self.min_experts, self.n_experts)

        expert_indices = torch.full(
            (batch_size, self.max_experts),
            -1,
            dtype=torch.long,
            device=probs.device,
        )
        expert_weights = torch.zeros(
            (batch_size, self.max_experts),
            dtype=probs.dtype,
            device=probs.device,
        )

        for row in range(batch_size):
            row_probs = probs[row]
            row_selected = torch.nonzero(selected_mask[row], as_tuple=False).flatten()
            if row_selected.numel() < self.min_experts:
                row_selected = torch.topk(row_probs, k=fallback_k, dim=-1).indices

            if row_selected.numel() > self.max_experts:
                keep = torch.topk(row_probs[row_selected], k=self.max_experts, dim=-1).indices
                row_selected = row_selected[keep]

            row_weights = row_probs[row_selected]
            weight_sum = row_weights.sum()
            if float(weight_sum) <= 0.0:
                row_weights = torch.full_like(row_weights, 1.0 / max(int(row_weights.numel()), 1))
            else:
                row_weights = row_weights / weight_sum

            count = int(row_selected.numel())
            expert_indices[row, :count] = row_selected
            expert_weights[row, :count] = row_weights
            self._expert_count_histogram[count] += 1
            self._total_experts_selected += count

        self._total_tokens_routed += batch_size
        batch_mean = probs.mean(dim=0).detach().to(dtype=torch.float32, device="cpu")
        self._thresholds = self.alpha * self._thresholds + (1.0 - self.alpha) * batch_mean
        return expert_indices, expert_weights

    @property
    def stats(self) -> Dict[str, object]:
        avg = self._total_experts_selected / self._total_tokens_routed if self._total_tokens_routed else 0.0
        histogram = {
            expert_count: self._expert_count_histogram.get(expert_count, 0)
            for expert_count in range(self.min_experts, self.max_experts + 1)
        }
        return {
            "avg_experts_per_token": avg,
            "threshold_values": self._thresholds.tolist(),
            "total_tokens_routed": self._total_tokens_routed,
            "expert_count_histogram": histogram,
        }

    def _to_probs(self, router_logits: torch.Tensor) -> torch.Tensor:
        if router_logits.ndim == 1:
            router_logits = router_logits.unsqueeze(0)
        if router_logits.shape[-1] != self.n_experts:
            raise ValueError(
                f"Expected logits with last dimension {self.n_experts}, got {tuple(router_logits.shape)}"
            )
        return torch.softmax(router_logits.float(), dim=-1)
