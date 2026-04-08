from __future__ import annotations

from collections import OrderedDict

import torch

from outlier_engine.paging import ExpertPageManager, _ExpertWeights


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
