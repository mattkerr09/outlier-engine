"""
Tests for RoE (Routing on Ensemble) — OUTLIER-ENGINE-ROE-001.

Three tests:
  1. test_roe_k2_matches_baseline  — k=2 RoE path produces identical output to
                                     the standard k=2 path (no cached extras exist).
  2. test_roe_k4_runs              — k=4 RoE path runs without error and returns
                                     the correct output shape.
  3. test_roe_cache_only           — extras are drawn ONLY from cached experts;
                                     an expert not in the cache is never included.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from outlier_engine.paging import (
    ExpertPageManager,
    _ExpertWeights,
    _roe_augment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = 8   # hidden_dim
_I = 16  # intermediate_dim
_E = 8   # n_experts


def _random_ternary(shape, seed=0) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randint(-1, 2, shape, dtype=torch.int8, generator=rng)


def _make_expert(*, hidden=_D, intermediate=_I, seed=0) -> _ExpertWeights:
    gate_w = _random_ternary((intermediate, hidden), seed=seed)
    gate_s = torch.ones(intermediate, 1, dtype=torch.float16)
    up_w = _random_ternary((intermediate, hidden), seed=seed + 1)
    up_s = torch.ones(intermediate, 1, dtype=torch.float16)
    down_w = _random_ternary((hidden, intermediate), seed=seed + 2)
    down_s = torch.ones(hidden, 1, dtype=torch.float16)
    return _ExpertWeights(gate_w, gate_s, up_w, up_s, down_w, down_s)


def _uniform_probs(n_experts: int, top_ids: list[int]) -> torch.Tensor:
    """Return [1, n_experts] softmax probs where top_ids dominate."""
    logits = torch.zeros(1, n_experts)
    for eid in top_ids:
        logits[0, eid] += 3.0
    return F.softmax(logits, dim=-1)


def _selected_idx(top_ids: list[int]) -> torch.Tensor:
    return torch.tensor([top_ids], dtype=torch.long)


# ===========================================================================
# test_roe_k2_matches_baseline
# ===========================================================================

def test_roe_k2_matches_baseline():
    """
    When roe_top_k=2 and only 2 experts exist in cache, _roe_augment returns
    no extras — so the output is identical to the standard top-2 path.
    """
    probs = _uniform_probs(_E, [0, 1])
    selected_idx = _selected_idx([0, 1])

    # Cache contains exactly the two required experts — no room for extras
    cached_ids = {0, 1}

    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=2,  # == n_required → no extras possible
        cached_ids=cached_ids,
        single_token=True,
    )

    assert extras == [], f"Expected no extras for roe_top_k==top_k, got {extras}"
    assert roe_w is None, "Expected None weights when no extras are added"


def test_roe_k2_baseline_multi_token():
    """_roe_augment is a no-op for multi-token (prefill) inputs."""
    probs = _uniform_probs(_E, [0, 1])
    selected_idx = _selected_idx([0, 1])

    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=4,
        cached_ids={0, 1, 2, 3},
        single_token=False,  # prefill → always no-op
    )

    assert extras == []
    assert roe_w is None


# ===========================================================================
# test_roe_k4_runs
# ===========================================================================

def test_roe_k4_runs():
    """
    With roe_top_k=4 and 2 extra experts in cache, _roe_augment returns 2 extras
    and a weight tensor of length 4 that sums to 1.
    """
    top_ids = [0, 1]
    probs = _uniform_probs(_E, top_ids)
    selected_idx = _selected_idx(top_ids)

    # Cache has experts 2 and 3 available as extras
    cached_ids = {0, 1, 2, 3}

    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=4,
        cached_ids=cached_ids,
        single_token=True,
    )

    assert len(extras) == 2, f"Expected 2 extras, got {extras}"
    # All extras must be from cached_ids and not in required set
    for eid in extras:
        assert eid in cached_ids, f"Extra {eid} not in cache"
        assert eid not in set(top_ids), f"Extra {eid} duplicates a required expert"

    assert roe_w is not None
    assert roe_w.shape == (4,), f"Expected weights shape (4,), got {roe_w.shape}"
    assert abs(float(roe_w.sum()) - 1.0) < 1e-5, f"Weights don't sum to 1: {roe_w}"


def test_roe_k4_partial_cache():
    """When fewer extras are cached than requested, _roe_augment returns what's available."""
    top_ids = [0, 1]
    probs = _uniform_probs(_E, top_ids)
    selected_idx = _selected_idx(top_ids)

    # Only one extra expert (2) in cache — can't reach k=4
    cached_ids = {0, 1, 2}

    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=4,
        cached_ids=cached_ids,
        single_token=True,
    )

    assert extras == [2], f"Expected [2], got {extras}"
    assert roe_w is not None
    assert roe_w.shape == (3,), f"Expected (3,), got {roe_w.shape}"
    assert abs(float(roe_w.sum()) - 1.0) < 1e-5


# ===========================================================================
# test_roe_cache_only
# ===========================================================================

def test_roe_cache_only():
    """
    Extras are drawn ONLY from cached_ids.  An expert with high routing
    probability that is NOT in the cache must never appear in extras.

    _roe_augment searches within the top-roe_top_k candidates by probability.
    With roe_top_k=4, the search covers [0, 1, 5, 2].  Expert 5 is skipped
    because it is not in the cache; expert 2 is picked.
    """
    # Expert 5 has the third-highest probability (after required [0, 1]),
    # but is NOT in the cache — it must be skipped.
    # Expert 2 is fourth-highest and IS cached — it should be the extra.
    logits = torch.zeros(1, _E)
    logits[0, 0] = 5.0   # required
    logits[0, 1] = 5.0   # required
    logits[0, 5] = 4.5   # 3rd-highest probability but NOT cached
    logits[0, 2] = 4.0   # 4th-highest probability and cached
    probs = F.softmax(logits, dim=-1)
    selected_idx = _selected_idx([0, 1])

    cached_ids = {0, 1, 2}  # expert 5 deliberately excluded

    # roe_top_k=4 → search top-4 candidates: [0, 1, 5, 2]
    # expert 5 skipped (not cached), expert 2 added
    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=4,
        cached_ids=cached_ids,
        single_token=True,
    )

    assert 5 not in extras, f"Expert 5 (not cached) must not appear in extras: {extras}"
    assert extras == [2], f"Expected [2] (only cached non-required expert), got {extras}"
    assert roe_w is not None
    assert abs(float(roe_w.sum()) - 1.0) < 1e-5


def test_roe_cache_only_empty_cache():
    """When cache has no extras beyond required experts, _roe_augment returns nothing."""
    probs = _uniform_probs(_E, [0, 1])
    selected_idx = _selected_idx([0, 1])

    # Cache only has the required experts — no eligible extras
    cached_ids = {0, 1}

    extras, roe_w = _roe_augment(
        probs=probs,
        selected_idx=selected_idx,
        roe_top_k=4,
        cached_ids=cached_ids,
        single_token=True,
    )

    assert extras == []
    assert roe_w is None


# ===========================================================================
# ExpertPageManager.enable_roe integration
# ===========================================================================

def test_enable_roe_sets_attribute():
    """enable_roe() correctly sets roe_top_k on ExpertPageManager."""
    from collections import OrderedDict
    import threading

    mgr = ExpertPageManager.__new__(ExpertPageManager)
    # Minimal init without disk access
    mgr.model_dir = None
    mgr.device = torch.device("cpu")
    mgr.n_experts = _E
    mgr.n_layers = 2
    mgr.top_k = 2
    mgr.roe_top_k = 0
    mgr.max_experts_in_memory = 4
    mgr.max_warm_cache = 16
    mgr._debug = False
    mgr._forward_passes = 0
    mgr._lock = threading.RLock()
    mgr._hot_cache = OrderedDict()
    mgr._cpu_cache = OrderedDict()
    mgr._hot_hits = 0
    mgr._warm_hits = 0
    mgr._cold_misses = 0
    mgr._hot_evictions = 0
    mgr._disk_loads = 0
    mgr._disk_load_s = 0.0
    mgr._usage_tracker = None
    mgr._pinned = frozenset()
    mgr._tensor_shard_index = {}
    mgr._packed_index = {}
    mgr._packed_experts_dir = None
    mgr._monolith_loader = None
    mgr._monolith_path = None
    mgr._fmt = "real"
    mgr._prefetcher = None
    mgr._routing_predictor = None
    mgr._last_routed_layer = None
    mgr._last_routed_experts = []
    mgr._et_routers = None
    mgr._cache_prior_routers = None
    mgr._debug_log = lambda msg: None

    assert mgr.roe_top_k == 0

    mgr.enable_roe(4)
    assert mgr.roe_top_k == 4

    mgr.enable_roe(0)
    assert mgr.roe_top_k == 0

    mgr.enable_roe(-1)  # negative should clamp to 0
    assert mgr.roe_top_k == 0
