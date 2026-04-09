"""
Tests for paging v2 integration: monolith loading, batched GEMM parity,
alpha scaling in the batched path, and graceful fallback.

OUTLIER-ENGINE-BATCHED-GEMM-001 / OUTLIER-ENGINE-MONOLITH-001
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from outlier_engine.paging import (
    ExpertPageManager,
    MonolithExpertLoader,
    _ExpertWeights,
    _resolve_monolith_path,
    _run_single_token_experts_batched,
    pack_ternary_tq10,
    unpack_ternary_tq10,
)
from outlier_engine.expert_store import ExpertStore, SUB_FILE_ORDER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = 8   # hidden_dim
_I = 16  # intermediate_dim


def _random_ternary(shape, seed=0) -> torch.Tensor:
    """Deterministic {-1, 0, +1} tensor."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    raw = torch.randint(-1, 2, shape, dtype=torch.int8, generator=rng)
    return raw


def _make_expert_weights(*, hidden=_D, intermediate=_I, seed=0) -> _ExpertWeights:
    """Unpacked int8 expert for CPU tests."""
    gate_w = _random_ternary((intermediate, hidden), seed=seed)
    gate_s = torch.ones(intermediate, 1, dtype=torch.float16)
    up_w = _random_ternary((intermediate, hidden), seed=seed + 1)
    up_s = torch.ones(intermediate, 1, dtype=torch.float16)
    down_w = _random_ternary((hidden, intermediate), seed=seed + 2)
    down_s = torch.ones(hidden, 1, dtype=torch.float16)
    return _ExpertWeights(gate_w, gate_s, up_w, up_s, down_w, down_s)


def _make_packed_expert(
    *,
    packed_dir: Path,
    layer: int,
    expert: int,
    hidden: int = _D,
    intermediate: int = _I,
    seed: int = 0,
) -> dict:
    """Write one TQ1_0-packed expert to packed_dir, return index entries."""
    ew = _make_expert_weights(hidden=hidden, intermediate=intermediate, seed=seed)
    index_entries = {}
    for proj, (w, s) in [
        ("gate", (ew.gate_w, ew.gate_s)),
        ("up", (ew.up_w, ew.up_s)),
        ("down", (ew.down_w, ew.down_s)),
    ]:
        # ternary
        packed = pack_ternary_tq10(w)
        ternary_filename = f"base_model_layers_{layer}_mlp_experts_{expert}_{proj}_ternary.bin"
        (packed_dir / ternary_filename).write_bytes(packed.numpy().tobytes())
        ternary_key = f"base.model.layers.{layer}.mlp.experts.{expert}.{proj}_ternary"
        index_entries[ternary_key] = {
            "file": ternary_filename,
            "shape": list(w.shape),
            "dtype": "uint8",
            "format": "tq10",
            "packed_bytes": packed.numel(),
        }
        # scale
        scale_filename = f"base_model_layers_{layer}_mlp_experts_{expert}_{proj}_scale.bin"
        scale_arr = s.numpy().astype(np.float16)
        (packed_dir / scale_filename).write_bytes(scale_arr.tobytes())
        scale_key = f"base.model.layers.{layer}.mlp.experts.{expert}.{proj}_scale"
        index_entries[scale_key] = {
            "file": scale_filename,
            "shape": list(s.shape),
            "dtype": "float16",
            "format": "raw",
            "packed_bytes": scale_arr.nbytes,
        }
    return index_entries


def _build_monolith(packed_dir: Path, output_path: Path) -> dict:
    """Create index.json from all .bin files then pack into monolith."""
    # Assemble index.json from all entries we've written
    entries = {}
    for f in packed_dir.glob("*.bin"):
        # Parse filename to reconstruct key
        pass  # index.json already written by caller
    return ExpertStore.pack(packed_dir, output_path)


def _make_manager_from_dir(packed_dir: Path, *, monolith_path=None) -> ExpertPageManager:
    """Build a bare ExpertPageManager with packed_index but no real model dir."""
    mgr = ExpertPageManager.__new__(ExpertPageManager)
    mgr.model_dir = packed_dir
    mgr.device = torch.device("cpu")
    mgr.n_experts = 8
    mgr.n_layers = 1
    mgr.top_k = 2
    mgr.max_experts_in_memory = 16
    mgr.max_warm_cache = 64
    mgr._debug = False
    mgr._forward_passes = 0
    mgr._lock = __import__("threading").RLock()
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
    mgr._fmt = "real"
    mgr._prefetcher = None
    mgr._routing_predictor = None
    mgr._last_routed_layer = None
    mgr._last_routed_experts = []
    mgr._et_routers = None
    mgr._cache_prior_routers = None
    mgr._packed_experts_dir = packed_dir
    mgr._monolith_loader = None
    mgr._monolith_path = None
    mgr._debug_log = lambda msg: None

    # Load packed index
    idx_path = packed_dir / "index.json"
    mgr._packed_index = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    # Initialise monolith loader if requested
    if monolith_path is not None:
        mgr._init_monolith_loader(monolith_path)

    return mgr


# ===========================================================================
# test_monolith_load
# ===========================================================================

def test_monolith_load():
    """
    Monolith experts.bin loads the same weights as the individual packed files.
    Verifies:
      - MonolithExpertLoader can open and parse experts.bin
      - Loaded _ExpertWeights (gate/up/down) match per-file loading byte-for-byte
        after unpacking TQ1_0 → int8
    """
    with tempfile.TemporaryDirectory() as tmp:
        packed_dir = Path(tmp)

        # Write two experts across one layer
        index = {}
        index.update(_make_packed_expert(packed_dir=packed_dir, layer=0, expert=0, seed=10))
        index.update(_make_packed_expert(packed_dir=packed_dir, layer=0, expert=1, seed=20))
        (packed_dir / "index.json").write_text(json.dumps(index))

        monolith_path = packed_dir / "experts.bin"
        ExpertStore.pack(packed_dir, monolith_path)

        loader = MonolithExpertLoader(monolith_path, index)

        for expert_id, seed in [(0, 10), (1, 20)]:
            from_monolith = loader.load_expert(0, expert_id)
            assert from_monolith.packed is True
            assert from_monolith.packed_format == "tq10"

            # Unpack and compare against original int8 weights
            unpacked = from_monolith.unpack_to_int8()
            original = _make_expert_weights(seed=seed)

            assert torch.equal(unpacked.gate_w, original.gate_w), \
                f"expert {expert_id}: gate_w mismatch"
            assert torch.equal(unpacked.up_w, original.up_w), \
                f"expert {expert_id}: up_w mismatch"
            assert torch.equal(unpacked.down_w, original.down_w), \
                f"expert {expert_id}: down_w mismatch"

        loader.close()


# ===========================================================================
# test_batched_gemm_parity
# ===========================================================================

def test_batched_gemm_parity():
    """
    Batched GEMM path produces outputs identical to the sequential per-expert path
    within bfloat16 numerical tolerance.

    OUTLIER-ENGINE-BATCHED-GEMM-001: 4 kernel launches for n experts.
    """
    torch.manual_seed(42)
    hidden = 16
    intermediate = 32
    n_experts = 4
    x = torch.randn(1, hidden, dtype=torch.bfloat16)

    experts = []
    for i in range(n_experts):
        ew = _make_expert_weights(hidden=hidden, intermediate=intermediate, seed=i)
        hot = ew.hot_ready(torch.device("cpu"))
        experts.append(hot)

    # Sequential baseline
    seq_outs = [ew.run(x) for ew in experts]
    seq_stack = torch.cat(seq_outs, dim=0)  # [E, H]

    # Batched path
    batched = _run_single_token_experts_batched(x, experts)

    assert batched.shape == (n_experts, hidden)
    # bfloat16 matmul — exact equality on CPU
    assert torch.allclose(batched, seq_stack, atol=1e-2, rtol=1e-2), (
        f"max abs diff: {(batched - seq_stack).abs().max().item():.4f}"
    )


# ===========================================================================
# test_alpha_batched
# ===========================================================================

def test_alpha_batched():
    """
    Alpha scaling applied in the batched path produces the same result as
    scaling each expert's output individually in the sequential path.

    Validates the weighted-combine step in _HybridPagedMLP.forward():
      expert_out[0] = (batched_out * (w_vec * alpha_vec).unsqueeze(-1)).sum(dim=0)
    """
    torch.manual_seed(99)
    hidden = 8
    intermediate = 16
    n_experts = 3
    x = torch.randn(1, hidden, dtype=torch.bfloat16)

    alphas = {0: 0.5, 1: 1.0, 2: 2.0}
    weights = {0: 0.4, 1: 0.35, 2: 0.25}  # routing weights (sum=1)

    experts = []
    for i in range(n_experts):
        ew = _make_expert_weights(hidden=hidden, intermediate=intermediate, seed=i * 7)
        hot = ew.hot_ready(torch.device("cpu"))
        experts.append(hot)

    # Reference: sequential with manual alpha scaling (accumulate in float32)
    ref_out = torch.zeros(hidden, dtype=torch.float32)
    for i, ew in enumerate(experts):
        out = ew.run(x).squeeze(0).float()  # [H] in float32
        ref_out += weights[i] * alphas[i] * out

    # Batched path (mirrors _HybridPagedMLP.forward logic)
    batched_out = _run_single_token_experts_batched(x, experts)  # [E, H]
    used_expert_ids = list(range(n_experts))

    w_vec = torch.tensor([weights[i] for i in used_expert_ids], dtype=torch.float32)
    alpha_vec = torch.tensor([alphas[i] for i in used_expert_ids], dtype=torch.float32)
    batched_result = (batched_out.float() * (w_vec * alpha_vec).unsqueeze(-1)).sum(dim=0)  # [H]

    assert torch.allclose(batched_result, ref_out, atol=1e-2, rtol=1e-2), (
        f"max abs diff: {(batched_result - ref_out).abs().max().item():.4f}"
    )


# ===========================================================================
# test_fallback
# ===========================================================================

def test_fallback():
    """
    When no monolith file (experts.bin) exists, ExpertPageManager._load_expert_from_disk
    silently falls through to the packed individual-file path.

    Also verifies that _resolve_monolith_path returns None for missing paths and
    that MonolithExpertLoader is NOT initialised in that case.
    """
    with tempfile.TemporaryDirectory() as tmp:
        packed_dir = Path(tmp)

        # Write a single expert using packed per-file format
        index = {}
        index.update(_make_packed_expert(packed_dir=packed_dir, layer=0, expert=0, seed=77))
        (packed_dir / "index.json").write_text(json.dumps(index))

        # No experts.bin exists here
        assert not (packed_dir / "experts.bin").exists()

        # _resolve_monolith_path must return None
        resolved = _resolve_monolith_path(packed_dir, packed_dir, None)
        assert resolved is None, f"Expected None, got {resolved}"

        # Build manager with no monolith — should NOT raise
        mgr = _make_manager_from_dir(packed_dir, monolith_path=None)
        assert mgr._monolith_loader is None, "Monolith loader should be None when no experts.bin"

        # get_expert must still work via the packed per-file fallback
        result = mgr.get_expert(0, 0)
        assert result.dequantized is True
        assert result.gate_w.shape[1] == _D
