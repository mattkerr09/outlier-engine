from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence

import torch


class RoutingPredictor:
    """Predict next-layer experts from current-layer routing patterns."""

    def __init__(self, *, warmup_updates: int = 4) -> None:
        self.warmup_updates = warmup_updates
        self._transition_counts: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
        self._updates = 0

    @staticmethod
    def _normalize_expert_ids(expert_ids: Iterable[int] | Sequence[int] | torch.Tensor) -> list[int]:
        if isinstance(expert_ids, torch.Tensor):
            values = expert_ids.detach().view(-1).tolist()
        else:
            values = list(expert_ids)
        unique: list[int] = []
        seen: set[int] = set()
        for value in values:
            expert_id = int(value)
            if expert_id not in seen:
                seen.add(expert_id)
                unique.append(expert_id)
        return unique

    def update(
        self,
        layer_idx: int,
        current_expert_ids: Iterable[int] | torch.Tensor,
        next_expert_ids: Iterable[int] | torch.Tensor,
    ) -> None:
        current = self._normalize_expert_ids(current_expert_ids)
        nxt = self._normalize_expert_ids(next_expert_ids)
        if not current or not nxt:
            return

        layer_table = self._transition_counts[layer_idx]
        for current_expert in current:
            counts = layer_table[current_expert]
            for next_expert in nxt:
                counts[next_expert] += 1
        self._updates += 1

    def predict(
        self,
        layer_idx: int,
        current_expert_ids: Iterable[int] | torch.Tensor,
        *,
        top_k: int,
    ) -> list[int]:
        current = self._normalize_expert_ids(current_expert_ids)
        if not current or top_k <= 0:
            return []

        layer_table = self._transition_counts.get(layer_idx)
        if self._updates < self.warmup_updates or not layer_table:
            return current[:top_k]

        aggregate: Counter[int] = Counter()
        for current_expert in current:
            aggregate.update(layer_table.get(current_expert, Counter()))

        predicted = [expert_id for expert_id, _count in aggregate.most_common(top_k)]
        if len(predicted) < top_k:
            for expert_id in current:
                if expert_id not in predicted:
                    predicted.append(expert_id)
                if len(predicted) >= top_k:
                    break
        return predicted[:top_k]
