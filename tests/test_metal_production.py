"""
Pytest tests for the production Metal TQ1_0 GEMV kernel.
Delegates to the canonical test implementations in outlier_engine/metal/.
"""

import pytest
import numpy as np

from outlier_engine.metal.metal_runtime import metal_available, TernaryMetal
from outlier_engine.metal.test_metal_kernel import (
    pack_matrix_tq1,
    ref_tq1_gemv,
    ref_fused_expert,
)

pytestmark = pytest.mark.skipif(
    not metal_available(), reason="metalcompute not available"
)


@pytest.fixture(scope="module")
def tm():
    return TernaryMetal()


def test_gemv_small(tm):
    rng = np.random.default_rng(1)
    M, D = 128, 64
    W = rng.integers(-1, 2, size=(M, D), dtype=np.int8)
    x = rng.standard_normal(D).astype(np.float16)
    packed, scales = pack_matrix_tq1(W)
    ref = ref_tq1_gemv(x, W, scales.reshape(M, -1))
    got = tm.gemv(x, packed, scales, M, D)
    max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_err < 1e-3, f"max_err={max_err}"


def test_gemv_real_dims(tm):
    rng = np.random.default_rng(2)
    M, D = 128, 3584
    W = rng.integers(-1, 2, size=(M, D), dtype=np.int8)
    x = rng.standard_normal(D).astype(np.float16)
    packed, scales = pack_matrix_tq1(W)
    ref = ref_tq1_gemv(x, W, scales.reshape(M, -1))
    got = tm.gemv(x, packed, scales, M, D)
    max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_err < 0.5, f"max_err={max_err}"


def test_fused_expert_small(tm):
    rng = np.random.default_rng(3)
    D, I = 64, 256
    gate_W = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    up_W   = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    down_W = rng.integers(-1, 2, size=(D, I), dtype=np.int8)
    x      = rng.standard_normal(D).astype(np.float16)
    gp, gs = pack_matrix_tq1(gate_W)
    up2, us = pack_matrix_tq1(up_W)
    dp, ds  = pack_matrix_tq1(down_W)
    ref = ref_fused_expert(x, gate_W, up_W, down_W)
    got = tm.fused_expert(x, gp, gs, up2, us, dp, ds, D, I)
    max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_err < 2.0, f"max_err={max_err}"


def test_fused_expert_medium(tm):
    rng = np.random.default_rng(4)
    D, I = 256, 1024
    gate_W = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    up_W   = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    down_W = rng.integers(-1, 2, size=(D, I), dtype=np.int8)
    x      = rng.standard_normal(D).astype(np.float16)
    gp, gs = pack_matrix_tq1(gate_W)
    up2, us = pack_matrix_tq1(up_W)
    dp, ds  = pack_matrix_tq1(down_W)
    ref = ref_fused_expert(x, gate_W, up_W, down_W)
    got = tm.fused_expert(x, gp, gs, up2, us, dp, ds, D, I)
    max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_err < 5.0, f"max_err={max_err}"


def test_scale_applied(tm):
    rng = np.random.default_rng(5)
    M, D = 32, 64
    W = np.ones((M, D), dtype=np.int8)
    x = np.ones(D, dtype=np.float16)
    scales_per_row = (D + 31) // 32
    packed, _ = pack_matrix_tq1(W)
    scales = np.zeros(M * scales_per_row, dtype=np.float16)
    for r in range(M):
        for g in range(scales_per_row):
            scales[r * scales_per_row + g] = float(r + 1)
    got = tm.gemv(x, packed, scales, M, D)
    expected = np.array([D * (r + 1) for r in range(M)], dtype=np.float32)
    rel_err = np.abs(got.astype(np.float32) - expected) / (np.abs(expected) + 1e-6)
    assert rel_err.max() < 0.01, f"max_rel_err={rel_err.max()}"


def test_fused_expert_preloaded(tm):
    """Pre-loaded expert (production path) gives identical result to one-shot."""
    rng = np.random.default_rng(6)
    D, I = 64, 256
    gate_W = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    up_W   = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    down_W = rng.integers(-1, 2, size=(D, I), dtype=np.int8)
    x      = rng.standard_normal(D).astype(np.float16)
    gp, gs = pack_matrix_tq1(gate_W)
    up2, us = pack_matrix_tq1(up_W)
    dp, ds  = pack_matrix_tq1(down_W)
    one_shot  = tm.fused_expert(x, gp, gs, up2, us, dp, ds, D, I)
    expert    = tm.load_expert(gp, gs, up2, us, dp, ds, D, I)
    preloaded = expert.forward(x)
    max_err   = float(np.abs(one_shot.astype(np.float32) - preloaded.astype(np.float32)).max())
    assert max_err < 1e-4, f"one_shot vs preloaded diverged: max_err={max_err}"
