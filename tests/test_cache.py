from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from outlier_engine.paging import ExpertPageManager, ExpertUsageTracker, _ExpertWeights


def _tiny_weights() -> _ExpertWeights:
    gate_w = torch.tensor([[1, 0], [0, -1]], dtype=torch.int8)
    gate_s = torch.ones(2, 1, dtype=torch.float16)
    up_w = torch.tensor([[0, 1], [1, 0]], dtype=torch.int8)
    up_s = torch.ones(2, 1, dtype=torch.float16)
    down_w = torch.tensor([[1, 0], [0, 1]], dtype=torch.int8)
    down_s = torch.ones(2, 1, dtype=torch.float16)
    return _ExpertWeights(gate_w, gate_s, up_w, up_s, down_w, down_s)


def _make_manager(*, hot_capacity: int = 2, warm_capacity: int = 4, device: str = "cpu") -> ExpertPageManager:
    mgr = ExpertPageManager.__new__(ExpertPageManager)
    mgr.model_dir = None
    mgr.device = torch.device(device)
    mgr.n_experts = 8
    mgr.max_experts_in_memory = hot_capacity
    mgr.max_warm_cache = warm_capacity
    mgr._debug = False
    mgr._forward_passes = 0
    mgr._hot_cache = OrderedDict()
    mgr._cpu_cache = OrderedDict()
    mgr._hot_hits = 0
    mgr._warm_hits = 0
    mgr._cold_misses = 0
    mgr._hot_evictions = 0
    mgr._disk_loads = 0
    mgr._disk_load_s = 0.0
    mgr._tensor_shard_index = {}
    mgr._packed_index = {}
    mgr._packed_experts_dir = None
    mgr._fmt = "real"
    mgr._debug_log = lambda message: None
    mgr._usage_tracker = None
    mgr._pinned = frozenset()
    return mgr


def test_hot_cache_lru_eviction(monkeypatch):
    mgr = _make_manager(hot_capacity=2, warm_capacity=8)
    monkeypatch.setattr(mgr, "_load_expert_from_disk", lambda layer, expert: _tiny_weights())

    mgr.get_expert(0, 0)
    mgr.get_expert(0, 1)
    mgr.get_expert(0, 2)

    assert (0, 0) not in mgr._hot_cache
    assert list(mgr._hot_cache.keys()) == [(0, 1), (0, 2)]
    assert mgr._hot_evictions == 1


def test_warm_cache_lru_eviction():
    mgr = _make_manager(hot_capacity=1, warm_capacity=2)

    mgr._add_to_warm_cache((0, 0), _tiny_weights())
    mgr._add_to_warm_cache((0, 1), _tiny_weights())
    mgr._add_to_warm_cache((0, 2), _tiny_weights())

    assert (0, 0) not in mgr._cpu_cache
    assert list(mgr._cpu_cache.keys()) == [(0, 1), (0, 2)]


def test_lookup_order():
    mgr = _make_manager(hot_capacity=2, warm_capacity=4)
    disk_calls: list[tuple[int, int]] = []

    hot = _tiny_weights().hot_ready(torch.device("cpu"))
    warm = _tiny_weights()
    mgr._hot_cache[(0, 0)] = hot
    mgr._cpu_cache[(0, 0)] = warm
    mgr._cpu_cache[(0, 1)] = warm

    def fake_disk(layer: int, expert: int) -> _ExpertWeights:
        disk_calls.append((layer, expert))
        return _tiny_weights()

    mgr._load_expert_from_disk = fake_disk

    hit_hot = mgr.get_expert(0, 0)
    hit_warm = mgr.get_expert(0, 1)
    miss_cold = mgr.get_expert(0, 2)

    assert hit_hot is hot
    assert hit_warm.dequantized is True
    assert miss_cold.dequantized is True
    assert disk_calls == [(0, 2)]


def test_hot_cache_hit_skips_unpack(monkeypatch):
    mgr = _make_manager(hot_capacity=2, warm_capacity=4)
    hot = _tiny_weights().hot_ready(torch.device("cpu"))
    mgr._hot_cache[(1, 1)] = hot

    def fail_unpack(self):
        raise AssertionError("unpack should not be called on hot hit")

    monkeypatch.setattr(_ExpertWeights, "unpack_to_int8", fail_unpack)
    mgr._load_expert_from_disk = lambda layer, expert: (_ for _ in ()).throw(AssertionError("disk should not be used"))

    returned = mgr.get_expert(1, 1)

    assert returned is hot
    assert mgr._hot_hits == 1


def test_cache_stats_tracking(monkeypatch):
    mgr = _make_manager(hot_capacity=2, warm_capacity=4)
    monkeypatch.setattr(mgr, "_load_expert_from_disk", lambda layer, expert: _tiny_weights())

    mgr.get_expert(0, 0)  # cold miss
    mgr.get_expert(0, 0)  # hot hit
    mgr._hot_cache.clear()
    mgr.get_expert(0, 0)  # warm hit

    stats = mgr.cache_stats()
    assert stats["hot_hits"] == 1
    assert stats["warm_hits"] == 1
    assert stats["cold_misses"] == 1
    assert stats["lookups"] == 3
    assert stats["hot_cache_entries"] >= 1


# ---------------------------------------------------------------------------
# ExpertUsageTracker tests
# ---------------------------------------------------------------------------

def test_usage_tracker_pins_after_threshold():
    tracker = ExpertUsageTracker(pin_after_tokens=3, pin_top_k=2)

    # Record expert accesses: (0,0) most frequent
    for _ in range(5):
        tracker.record(0, 0)
    for _ in range(2):
        tracker.record(0, 1)
    tracker.record(0, 2)

    # No pinning yet — threshold not reached
    assert tracker.pinned == frozenset()

    for _ in range(3):
        tracker.on_token()

    # After 3 tokens, top-2 should be pinned
    assert tracker.pinned == {(0, 0), (0, 1)}


def test_usage_tracker_pin_hit_rate():
    tracker = ExpertUsageTracker(pin_after_tokens=1, pin_top_k=1)
    tracker.record(0, 0)
    tracker.on_token()  # triggers pinning: (0,0) is pinned

    tracker.record(0, 0)  # pin hit
    tracker.record(0, 1)  # not pinned

    # 3 total lookups (including pre-pin record), 1 hit after pinning
    assert tracker.pin_hit_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Pinned-eviction integration tests
# ---------------------------------------------------------------------------

def test_pinned_expert_not_evicted(monkeypatch):
    """A pinned expert must survive when the hot cache is full."""
    mgr = _make_manager(hot_capacity=2, warm_capacity=8)
    monkeypatch.setattr(mgr, "_load_expert_from_disk", lambda layer, expert: _tiny_weights())

    mgr.enable_expert_pinning(pin_top_k=1, pin_after_tokens=2)

    # Access (0,0) many times to make it the top expert
    for _ in range(5):
        mgr.get_expert(0, 0)
    mgr.get_expert(0, 1)

    # Simulate 2 tokens to trigger pinning
    mgr.debug_forward_start("tok1")
    mgr.debug_forward_start("tok2")

    assert (0, 0) in mgr._pinned

    # Fill the hot cache, then add a new expert — (0,0) must NOT be evicted
    mgr.get_expert(0, 2)  # evicts LRU non-pinned entry
    assert (0, 0) in mgr._hot_cache


def test_cache_stats_reports_pinning(monkeypatch):
    mgr = _make_manager(hot_capacity=4, warm_capacity=8)
    monkeypatch.setattr(mgr, "_load_expert_from_disk", lambda layer, expert: _tiny_weights())

    # Without pinning enabled
    stats = mgr.cache_stats()
    assert stats["pinning_enabled"] is False
    assert stats["pinned_expert_count"] == 0

    # Enable pinning and trigger threshold
    mgr.enable_expert_pinning(pin_top_k=1, pin_after_tokens=1)
    mgr.get_expert(0, 0)
    mgr.debug_forward_start("tok1")  # triggers pinning

    stats = mgr.cache_stats()
    assert stats["pinning_enabled"] is True
    assert stats["pinned_expert_count"] == 1
