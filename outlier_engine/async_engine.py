"""
async_engine.py — Async expert prefetch with CUDA stream overlap.
OUTLIER-RUNTIME-004

Overlaps expert loading with attention computation using:
  - CUDA: two streams (compute + copy)
  - MPS/CPU: background thread with ThreadPoolExecutor

Architecture:

  Layer N:
    [Attention compute]  ──stream_compute──>  [MoE forward]
                         ──stream_copy──>  [Prefetch layer N+1 experts]

  The prefetch for layer N+1 starts as soon as layer N's router
  logits are available (before attention finishes — we run the router
  first on the pre-attention hidden state as an approximation).

References:
  ExpertFlow (2024): temporal locality in expert activation (~70–80% overlap)
  MoBiLE (2024): big-little routing reduces expert loads 1.6–1.7×
"""

from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Async Expert Prefetcher
# ---------------------------------------------------------------------------

class AsyncExpertPrefetcher:
    """
    Overlaps expert loading with attention computation using:
    - CUDA: two streams (compute + copy)
    - MPS/CPU: background thread with ThreadPoolExecutor

    Architecture:

    Layer N:
      [Attention compute]  ──stream_compute──>  [MoE forward]
                           ──stream_copy──>  [Prefetch layer N+1 experts]

    The prefetch for layer N+1 starts as soon as layer N's router
    logits are available (which happens BEFORE attention finishes
    if we run the router first).
    """

    def __init__(self, paged_model, max_prefetch_ahead: int = 2) -> None:
        """
        paged_model: OutlierPagedModel instance
        max_prefetch_ahead: how many layers ahead to prefetch (1–3)
        """
        self.model     = paged_model
        self.max_ahead = max_prefetch_ahead
        self.device    = next(paged_model.parameters()).device

        # CUDA streams for overlap
        if self.device.type == "cuda":
            self.stream_compute = torch.cuda.Stream()
            self.stream_copy    = torch.cuda.Stream()

        # CPU/MPS: use thread pool for async loading
        self.executor       = ThreadPoolExecutor(max_workers=2)
        self.pending_futures: Dict[Tuple[int, int], object] = {}

        # Prefetch cache: layer_idx -> {expert_id: _ExpertWeights}
        self.prefetch_buffer: Dict[int, Dict[int, object]] = {}

        # Stats
        self.stats = {
            "prefetch_hits":   0,
            "prefetch_misses": 0,
            "total_expert_loads": 0,
        }

    def predict_next_experts(
        self,
        layer_idx: int,
        router_logits: torch.Tensor,
        topk: int = 4,
    ) -> torch.Tensor:
        """
        Use current layer's router logits to predict next layer's experts.

        Simple heuristic (v0.4): assume same experts will be needed.
        This achieves ~70% accuracy based on ExpertFlow research.

        Future (v0.5): MoBiLE-style confidence-based prediction.
        Future (v0.6): lightweight RPP transformer.

        router_logits: (N, n_experts)  raw logits (pre-softmax)
        Returns: (N, topk)  predicted expert indices for layer_idx+1
        """
        k = min(topk, router_logits.shape[-1])
        _, topk_indices = torch.topk(router_logits, k=k, dim=-1)
        # Prediction: temporal locality — same experts for next layer
        return topk_indices

    def prefetch_experts_async(
        self,
        layer_idx: int,
        expert_ids: torch.Tensor,
    ) -> None:
        """
        Start loading experts for layer_idx in background.

        On CUDA: use stream_copy to transfer from CPU to GPU.
        On MPS:  use ThreadPoolExecutor (MPS doesn't support streams).
        On CPU:  no-op (everything is already in RAM).
        """
        if self.device.type == "cpu":
            return  # Nothing to prefetch

        expert_ids_list = expert_ids.flatten().unique().cpu().tolist()
        expert_ids_list = [int(e) for e in expert_ids_list]

        if self.device.type == "cuda":
            with torch.cuda.stream(self.stream_copy):
                for eid in expert_ids_list:
                    if eid not in self.prefetch_buffer.get(layer_idx, {}):
                        expert = self.model.load_expert(layer_idx, eid)
                        if expert is not None:
                            if layer_idx not in self.prefetch_buffer:
                                self.prefetch_buffer[layer_idx] = {}
                            self.prefetch_buffer[layer_idx][eid] = expert
                            self.stats["total_expert_loads"] += 1

        elif self.device.type == "mps":
            def _load_fn(lid: int, eid: int) -> Tuple[int, int, object]:
                expert = self.model.load_expert(lid, eid)
                return (lid, eid, expert)

            for eid in expert_ids_list:
                if eid not in self.prefetch_buffer.get(layer_idx, {}):
                    key = (layer_idx, eid)
                    if key not in self.pending_futures:
                        future = self.executor.submit(_load_fn, layer_idx, eid)
                        self.pending_futures[key] = future

    def get_expert(self, layer_idx: int, expert_id: int) -> object:
        """
        Get an expert, checking prefetch buffer first.

        Returns _ExpertWeights on the model's device.
        """
        # Check prefetch buffer (CUDA path + manual pre-population)
        if layer_idx in self.prefetch_buffer:
            buf = self.prefetch_buffer[layer_idx]
            if expert_id in buf:
                self.stats["prefetch_hits"] += 1
                return buf.pop(expert_id)

        # Check pending futures (MPS path)
        key = (layer_idx, expert_id)
        if key in self.pending_futures:
            future = self.pending_futures.pop(key)
            _lid, _eid, expert = future.result()
            if expert is not None:
                self.stats["prefetch_hits"] += 1
                return expert

        # Cache miss — synchronous load
        self.stats["prefetch_misses"] += 1
        return self.model.load_expert(layer_idx, expert_id)

    def sync_prefetch(self) -> None:
        """Wait for all pending prefetch operations to complete."""
        if self.device.type == "cuda" and hasattr(self, "stream_copy"):
            self.stream_copy.synchronize()
        for key in list(self.pending_futures.keys()):
            future = self.pending_futures.pop(key)
            _lid, _eid, expert = future.result()
            if expert is not None:
                lid = key[0]
                eid = key[1]
                if lid not in self.prefetch_buffer:
                    self.prefetch_buffer[lid] = {}
                self.prefetch_buffer[lid][eid] = expert

    def clear_layer(self, layer_idx: int) -> None:
        """Clear prefetch buffer for a completed layer."""
        self.prefetch_buffer.pop(layer_idx, None)

    def report_stats(self) -> Dict[str, object]:
        total    = self.stats["prefetch_hits"] + self.stats["prefetch_misses"]
        hit_rate = self.stats["prefetch_hits"] / max(total, 1) * 100
        return {
            "prefetch_hit_rate":  f"{hit_rate:.1f}%",
            "total_loads":        self.stats["total_expert_loads"],
            "hits":               self.stats["prefetch_hits"],
            "misses":             self.stats["prefetch_misses"],
        }


# ---------------------------------------------------------------------------
# Async Forward Engine
# ---------------------------------------------------------------------------

class AsyncForwardEngine:
    """
    Modified forward pass that overlaps attention with expert prefetch.

    Standard forward (current, sequential):
      for layer in layers:
        x = attention(x)
        experts = load_experts(router(x))  # BLOCKS
        x = moe_forward(x, experts)

    Async forward (v0.4):
      for layer in layers:
        router_logits = router(x)           # pre-attention approximation
        prefetch.start(layer+1, predict(router_logits))  # NON-BLOCKING
        x = attention(x)                    # OVERLAPPED with prefetch
        prefetch.sync()                     # wait if prefetch not done
        experts = prefetch.get(layer, topk(router_logits))
        x = moe_forward(x, experts)
        prefetch.clear(layer)
    """

    def __init__(
        self,
        paged_model,
        max_prefetch_ahead: int = 2,
        big_little_router: Optional["BigLittleRouter"] = None,
    ) -> None:
        self.model            = paged_model
        self.prefetcher       = AsyncExpertPrefetcher(paged_model, max_prefetch_ahead)
        self.big_little_router: Optional[BigLittleRouter] = big_little_router

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Run full model forward with async expert prefetching.
        Returns logits tensor [B, L, vocab_size].
        """
        from .model import _causal_mask
        from .paging import _run_expert, _PagedMoEFFN

        m = self.model
        x = m.embed_tokens(input_ids)   # [B, L, D]
        B, L, D = x.shape

        causal_mask: Optional[torch.Tensor] = None
        if L > 1:
            causal_mask = _causal_mask(L, device=input_ids.device, dtype=x.dtype)

        for i, layer in enumerate(m.layers):

            # ── Router-first: approximate routing on pre-attention x ──────
            # Running the router BEFORE attention lets us start prefetching
            # current and next layer experts while attention computes.
            if layer.n_experts > 0:
                ffn: _PagedMoEFFN = layer.ffn  # type: ignore[assignment]

                h_pre      = layer.ffn_norm(x)
                h_pre_flat = h_pre.view(-1, D)
                logits_pre = F.linear(h_pre_flat.float(), ffn.router_weight.float())

                # Determine k for prefetch (big-little or full top-k)
                if self.big_little_router is not None:
                    k_prefetch = self.big_little_router.decide_topk(logits_pre[0])
                else:
                    k_prefetch = m.top_k

                k_prefetch = min(k_prefetch, m.n_experts)
                _, top_idx_pre = torch.topk(
                    F.softmax(logits_pre, dim=-1), k=k_prefetch, dim=-1
                )

                # Kick off async load for current layer's likely experts
                self.prefetcher.prefetch_experts_async(i, top_idx_pre)

                # Predict and prefetch NEXT layer's experts
                if i + 1 < m.n_layers:
                    next_pred = self.prefetcher.predict_next_experts(
                        i, logits_pre, topk=k_prefetch
                    )
                    self.prefetcher.prefetch_experts_async(i + 1, next_pred)

            # ── Attention (overlapped with expert prefetch) ───────────────
            x = x + layer.attn(layer.attn_norm(x), mask=causal_mask)

            # ── FFN ───────────────────────────────────────────────────────
            h = layer.ffn_norm(x)

            if layer.n_experts > 0:
                ffn = layer.ffn  # type: ignore[assignment]
                h_flat = h.view(-1, D)   # [N, D]
                N      = h_flat.shape[0]

                # Actual routing on post-attention x for accurate weights
                logits = F.linear(h_flat.float(), ffn.router_weight.float())
                probs  = F.softmax(logits, dim=-1)

                if self.big_little_router is not None:
                    k_actual = self.big_little_router.decide_topk(logits[0])
                    k_actual = min(k_actual, m.n_experts)
                else:
                    k_actual = m.top_k

                top_w, top_idx = torch.topk(probs, k=k_actual, dim=-1)
                top_w = top_w / top_w.sum(-1, keepdim=True)

                # Sync prefetch before using experts
                self.prefetcher.sync_prefetch()

                # Shared expert (always device-resident, INT8 quantised)
                shared_out = ffn.shared(h_flat).float()   # [N, D]

                # Ternary experts (from prefetch buffer or synchronous load)
                expert_out = torch.zeros(
                    N, D, device=m.device, dtype=torch.float32
                )
                for k_i in range(k_actual):
                    for e in range(m.n_experts):
                        mask_e = (top_idx[:, k_i] == e)
                        if mask_e.any():
                            w   = self.prefetcher.get_expert(i, e)
                            out = _run_expert(h_flat[mask_e], w)   # float32
                            expert_out[mask_e] += top_w[mask_e, k_i:k_i+1] * out

                self.prefetcher.clear_layer(i)
                x = x + (shared_out + expert_out).to(x.dtype).view(B, L, D)

            else:
                # Dense FFN — no paging needed
                x = x + layer.ffn(h)

        x = m.norm(x)
        return m.lm_head(x)   # [B, L, V]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with async prefetching.
        Reports tokens/sec and prefetch hit rate on completion.
        """
        tokens = input_ids
        t0     = time.time()

        for _ in range(max_new_tokens):
            logits      = self.forward(tokens)
            next_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs      = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if tokens.shape[1] >= self.model.max_seq_len:
                break

        elapsed  = time.time() - t0
        n_new    = tokens.shape[1] - input_ids.shape[1]
        tps      = n_new / max(elapsed, 1e-6)
        stats    = self.prefetcher.report_stats()
        print(
            f"Tokens/sec: {tps:.1f} | "
            f"Prefetch hit rate: {stats['prefetch_hit_rate']} | "
            f"Hits: {stats['hits']} / Misses: {stats['misses']}"
        )
        return tokens


# ---------------------------------------------------------------------------
# BigLittle Router — MoBiLE-style expert reduction (prep for v0.5)
# ---------------------------------------------------------------------------

class BigLittleRouter:
    """
    MoBiLE-inspired: use fewer experts for "easy" tokens.

    For ~80% of tokens, the router confidence for the top-K experts is high.
    These tokens only need top-4 (not top-8) experts — halving load bandwidth.

    Only when the 4th-to-8th expert scores are close to the top-4 scores
    (confidence gap < threshold) do we load all 8.

    Research says this achieves 1.6–1.7× speedup with negligible accuracy loss.
    """

    def __init__(
        self,
        full_topk: int = 8,
        little_topk: int = 4,
        confidence_threshold: float = 0.1,
    ) -> None:
        self.full_topk   = full_topk
        self.little_topk = little_topk
        self.threshold   = confidence_threshold
        self.stats       = {"little": 0, "big": 0}

    def decide_topk(self, router_logits: torch.Tensor) -> int:
        """
        Decide whether this token needs full top-K or reduced top-k.

        router_logits: (num_experts,)
        Returns: k (either little_topk or full_topk)
        """
        n_experts = router_logits.shape[0]
        # Need at least full_topk + 1 experts to compare the gap
        if n_experts <= self.little_topk:
            self.stats["little"] += 1
            return min(self.little_topk, n_experts)

        probs        = torch.softmax(router_logits.float(), dim=-1)
        sorted_probs = probs.sort(descending=True).values

        # Gap between little_topk-th and (little_topk+1)-th expert probability
        gap = (sorted_probs[self.little_topk - 1] - sorted_probs[self.little_topk]).item()

        if gap > self.threshold:
            # Clear separation — little experts are sufficient
            self.stats["little"] += 1
            return self.little_topk
        else:
            # Ambiguous — use full set
            self.stats["big"] += 1
            return min(self.full_topk, n_experts)

    def report(self) -> str:
        total      = self.stats["little"] + self.stats["big"]
        little_pct = self.stats["little"] / max(total, 1) * 100
        return f"Big-Little: {little_pct:.0f}% tokens used reduced experts"
