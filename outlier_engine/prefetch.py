from __future__ import annotations

import threading
from typing import Iterable, Optional, Sequence

import torch


class ExpertPrefetcher:
    """Thread-based expert prefetch controller for one-layer-ahead paging."""

    def __init__(self, expert_cache) -> None:
        self.expert_cache = expert_cache
        self._thread: Optional[threading.Thread] = None
        self._pending_layer: Optional[int] = None
        self._pending_experts: set[int] = set()
        self._issued_by_layer: dict[int, set[int]] = {}
        self._lock = threading.RLock()
        self._stats = {
            "prefetches_issued": 0,
            "prefetch_hits": 0,
            "prefetch_wastes": 0,
        }

    def _normalize_expert_ids(
        self,
        *,
        routing_logits: Optional[torch.Tensor] = None,
        top_k: int = 0,
        predicted_expert_ids: Optional[Sequence[int] | torch.Tensor] = None,
    ) -> list[int]:
        if predicted_expert_ids is not None:
            if isinstance(predicted_expert_ids, torch.Tensor):
                values = predicted_expert_ids.detach().view(-1).tolist()
            else:
                values = list(predicted_expert_ids)
            unique: list[int] = []
            seen: set[int] = set()
            for value in values:
                expert_id = int(value)
                if expert_id not in seen:
                    seen.add(expert_id)
                    unique.append(expert_id)
            return unique[:top_k] if top_k > 0 else unique

        if routing_logits is None or routing_logits.numel() == 0:
            return []

        k = min(int(top_k), int(routing_logits.shape[-1]))
        if k <= 0:
            return []
        _, top_idx = torch.topk(routing_logits.detach(), k=k, dim=-1)
        return [int(expert_id) for expert_id in top_idx.reshape(-1).unique().cpu().tolist()]

    def prefetch(
        self,
        layer_idx: int,
        *,
        routing_logits: Optional[torch.Tensor] = None,
        top_k: int,
        predicted_expert_ids: Optional[Sequence[int] | torch.Tensor] = None,
    ) -> list[int]:
        expert_ids = self._normalize_expert_ids(
            routing_logits=routing_logits,
            top_k=top_k,
            predicted_expert_ids=predicted_expert_ids,
        )
        if not expert_ids:
            return []

        self.wait()

        with self._lock:
            self._pending_layer = layer_idx
            self._pending_experts = set(expert_ids)
            self._issued_by_layer[layer_idx] = set(expert_ids)
            self._stats["prefetches_issued"] += len(expert_ids)

        def _target() -> None:
            for expert_id in expert_ids:
                self.expert_cache.prefetch_expert(layer_idx, int(expert_id))

        thread = threading.Thread(target=_target, name=f"expert-prefetch-{layer_idx}", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()
        return expert_ids

    def wait(self, layer_idx: Optional[int] = None) -> None:
        with self._lock:
            thread = self._thread
            pending_layer = self._pending_layer
        if thread is None:
            return
        if layer_idx is not None and pending_layer is not None and pending_layer != layer_idx:
            return
        thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._pending_layer = None
                self._pending_experts = set()

    def record_usage(self, layer_idx: int, used_expert_ids: Iterable[int] | torch.Tensor) -> None:
        if isinstance(used_expert_ids, torch.Tensor):
            used = {int(expert_id) for expert_id in used_expert_ids.detach().view(-1).tolist()}
        else:
            used = {int(expert_id) for expert_id in used_expert_ids}

        with self._lock:
            predicted = self._issued_by_layer.pop(layer_idx, None)
            if not predicted:
                return
            self._stats["prefetch_hits"] += len(predicted & used)
            self._stats["prefetch_wastes"] += len(predicted - used)

    @property
    def prefetch_stats(self) -> dict[str, float]:
        with self._lock:
            stats = dict(self._stats)
        issued = int(stats["prefetches_issued"])
        hits = int(stats["prefetch_hits"])
        stats["prefetch_accuracy"] = (hits / issued) if issued > 0 else 0.0
        return stats
