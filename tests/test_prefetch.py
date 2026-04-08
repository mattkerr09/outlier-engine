from __future__ import annotations

import time

import torch

from outlier_engine.prefetch import ExpertPrefetcher
from outlier_engine.routing_predictor import RoutingPredictor


class _FakeCache:
    def __init__(self) -> None:
        self.loaded: list[tuple[int, int]] = []

    def prefetch_expert(self, layer_idx: int, expert_idx: int) -> bool:
        time.sleep(0.001)
        self.loaded.append((layer_idx, expert_idx))
        return True


def test_expert_prefetcher_loads_background_and_tracks_stats():
    cache = _FakeCache()
    prefetcher = ExpertPrefetcher(cache)
    logits = torch.tensor([[0.1, 0.9, 0.3, 0.8]], dtype=torch.float32)

    prefetched = prefetcher.prefetch(2, routing_logits=logits, top_k=2)
    assert prefetched == [1, 3]

    prefetcher.wait(2)
    assert cache.loaded == [(2, 1), (2, 3)]

    prefetcher.record_usage(2, [1])
    stats = prefetcher.prefetch_stats
    assert stats["prefetches_issued"] == 2
    assert stats["prefetch_hits"] == 1
    assert stats["prefetch_wastes"] == 1
    assert stats["prefetch_accuracy"] == 0.5


def test_routing_predictor_falls_back_then_learns_transition():
    predictor = RoutingPredictor(warmup_updates=1)

    assert predictor.predict(0, [2, 4], top_k=2) == [2, 4]

    predictor.update(0, [2, 4], [5, 6])
    assert predictor.predict(0, [2], top_k=2) == [5, 6]
