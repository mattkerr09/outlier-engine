"""
test_metal_kernel.py — Correctness tests and benchmark for the production
TQ1_0 Metal GEMV kernel.

Run standalone:
  python outlier_engine/metal/test_metal_kernel.py
Run via pytest:
  pytest outlier_engine/metal/test_metal_kernel.py -v

Compares (at D=3584, I=18944 — Qwen2.5-7B / Outlier-10B expert dims):
  1. Production Metal — pre-loaded weights (production inference path)
  2. Production Metal — one-shot (numpy arrays copied each call)
  3. CPU fp16 matmul × 3 (torch)
  4. MPS fp16 matmul × 3 (torch on Apple GPU)
  5. Prototype naive Metal dequant-only
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from outlier_engine.metal.metal_runtime import TernaryMetal, metal_available

# ---------------------------------------------------------------------------
# Reference helpers (shared by tests and benchmark)
# ---------------------------------------------------------------------------

GROUP_SIZE = 32


def pack_tq1(weights_int: np.ndarray) -> np.ndarray:
    """Pack ternary weights {-1,0,1} into TQ1_0 bytes (5 per byte, base-3)."""
    n = len(weights_int)
    packed = []
    i = 0
    while i < n:
        b = 0
        for pos in range(5):
            val = int(weights_int[i + pos]) + 1 if i + pos < n else 0
            b += val * (3 ** pos)
        packed.append(b)
        i += 5
    return np.array(packed, dtype=np.uint8)


def pack_matrix_tq1(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack a [rows, cols] int8 matrix into TQ1_0. Unit scales (=1.0)."""
    rows, cols = W.shape
    bytes_per_row  = (cols + 4) // 5
    scales_per_row = (cols + 31) // 32
    packed = np.zeros(rows * bytes_per_row, dtype=np.uint8)
    scales = np.ones(rows * scales_per_row, dtype=np.float16)
    for r in range(rows):
        row_bytes = pack_tq1(W[r])
        packed[r * bytes_per_row: r * bytes_per_row + len(row_bytes)] = row_bytes
    return packed, scales


def ref_tq1_gemv(
    x:      np.ndarray,   # [D] float16
    W:      np.ndarray,   # [M, D] int8 ternary
    scales: np.ndarray,   # [M, scales_per_row] float16
) -> np.ndarray:          # [M] float16
    M, D = W.shape
    scales_per_row = (D + GROUP_SIZE - 1) // GROUP_SIZE
    W_fp = np.zeros((M, D), dtype=np.float32)
    for r in range(M):
        for g in range(scales_per_row):
            s = float(scales[r, g])
            start, end = g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, D)
            W_fp[r, start:end] = W[r, start:end].astype(np.float32) * s
    return (W_fp @ x.astype(np.float32)).astype(np.float16)


def ref_fused_expert(
    x:      np.ndarray,  # [D] float16
    gate_W: np.ndarray,  # [I, D] int8
    up_W:   np.ndarray,  # [I, D] int8
    down_W: np.ndarray,  # [D, I] int8
) -> np.ndarray:         # [D] float16
    x_t    = torch.from_numpy(x.astype(np.float32))
    gate_t = torch.from_numpy(gate_W.astype(np.float32))
    up_t   = torch.from_numpy(up_W.astype(np.float32))
    down_t = torch.from_numpy(down_W.astype(np.float32))
    gate_v = F.linear(x_t, gate_t)
    up_v   = F.linear(x_t, up_t)
    mid    = F.silu(gate_v) * up_v
    return F.linear(mid, down_t).to(torch.float16).numpy()


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

import pytest

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
    assert max_err < 1e-3, f"max_err={max_err:.6f}"


def test_gemv_real_dims(tm):
    rng = np.random.default_rng(2)
    M, D = 128, 3584
    W = rng.integers(-1, 2, size=(M, D), dtype=np.int8)
    x = rng.standard_normal(D).astype(np.float16)
    packed, scales = pack_matrix_tq1(W)
    ref = ref_tq1_gemv(x, W, scales.reshape(M, -1))
    got = tm.gemv(x, packed, scales, M, D)
    max_err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
    assert max_err < 0.5, f"max_err={max_err:.4f}"


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
    assert max_err < 2.0, f"max_err={max_err:.4f}"


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
    assert max_err < 5.0, f"max_err={max_err:.4f}"


def test_fused_expert_preloaded(tm):
    """Pre-loaded expert (production path) gives same result as one-shot."""
    rng = np.random.default_rng(6)
    D, I = 64, 256
    gate_W = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    up_W   = rng.integers(-1, 2, size=(I, D), dtype=np.int8)
    down_W = rng.integers(-1, 2, size=(D, I), dtype=np.int8)
    x      = rng.standard_normal(D).astype(np.float16)
    gp, gs = pack_matrix_tq1(gate_W)
    up2, us = pack_matrix_tq1(up_W)
    dp, ds  = pack_matrix_tq1(down_W)
    one_shot = tm.fused_expert(x, gp, gs, up2, us, dp, ds, D, I)
    expert   = tm.load_expert(gp, gs, up2, us, dp, ds, D, I)
    preloaded = expert.forward(x)
    max_err = float(np.abs(one_shot.astype(np.float32) - preloaded.astype(np.float32)).max())
    assert max_err < 1e-4, f"one_shot vs preloaded diverged: max_err={max_err}"


def test_scale_applied(tm):
    rng = np.random.default_rng(5)
    M, D = 32, 64
    W = np.ones((M, D), dtype=np.int8)
    x = np.ones(D, dtype=np.float16)
    scales_per_row = (D + GROUP_SIZE - 1) // GROUP_SIZE
    packed, _ = pack_matrix_tq1(W)
    scales = np.zeros(M * scales_per_row, dtype=np.float16)
    for r in range(M):
        for g in range(scales_per_row):
            scales[r * scales_per_row + g] = float(r + 1)
    got = tm.gemv(x, packed, scales, M, D)
    expected = np.array([D * (r + 1) for r in range(M)], dtype=np.float32)
    rel_err  = np.abs(got.astype(np.float32) - expected) / (np.abs(expected) + 1e-6)
    assert rel_err.max() < 0.01, f"max_rel_err={rel_err.max():.4f}"


# ---------------------------------------------------------------------------
# Standalone benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    if not metal_available():
        print("ERROR: metalcompute not available.")
        return

    tm = TernaryMetal()

    D_REAL = 3584
    I_REAL = 18944
    WARMUP = 30
    ITERS  = 300

    print("=" * 68)
    print("BENCHMARK  —  Production TQ1_0 Metal GEMV + Fused SwiGLU Expert")
    print(f"Dimensions: D={D_REAL}, I={I_REAL}  (Qwen2.5-7B / Outlier-10B expert)")
    print(f"Device: Apple M1 Ultra  |  warmup={WARMUP}  iters={ITERS}")
    print("=" * 68)

    rng = np.random.default_rng(0)
    bytes_gu  = (D_REAL + 4) // 5
    scales_gu = (D_REAL + 31) // 32
    bytes_dn  = (I_REAL + 4) // 5
    scales_dn = (I_REAL + 31) // 32

    x           = rng.standard_normal(D_REAL).astype(np.float16)
    gate_packed = rng.integers(0, 243, size=I_REAL * bytes_gu, dtype=np.uint8)
    gate_scales = (rng.standard_normal(I_REAL * scales_gu) * 0.01).astype(np.float16)
    up_packed   = rng.integers(0, 243, size=I_REAL * bytes_gu, dtype=np.uint8)
    up_scales   = (rng.standard_normal(I_REAL * scales_gu) * 0.01).astype(np.float16)
    down_packed = rng.integers(0, 243, size=D_REAL * bytes_dn, dtype=np.uint8)
    down_scales = (rng.standard_normal(D_REAL * scales_dn) * 0.01).astype(np.float16)

    bytes_tq10 = (
        I_REAL * bytes_gu + I_REAL * scales_gu * 2 +
        I_REAL * bytes_gu + I_REAL * scales_gu * 2 +
        D_REAL * bytes_dn + D_REAL * scales_dn * 2
    )
    bytes_fp16 = (I_REAL * D_REAL + I_REAL * D_REAL + D_REAL * I_REAL) * 2

    # ---- 0. Dispatch overhead baseline: empty noop kernel --------------------
    print("\n[0] Dispatch overhead — noop kernel (2 dispatches, no compute)")
    import metalcompute as mc
    dev0   = mc.Device()
    noop_src = open(os.path.join(os.path.dirname(__file__), "ternary_gemv.metal")).read()
    noop_fn = dev0.kernel(noop_src).function("noop")
    tiny_buf = dev0.buffer(4)
    for _ in range(WARMUP):
        noop_fn(1, tiny_buf)
        noop_fn(1, tiny_buf)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        noop_fn(1, tiny_buf)
        noop_fn(1, tiny_buf)
    t1 = time.perf_counter()
    ms_noop = (t1 - t0) / ITERS * 1000.0
    print(f"    {ms_noop:.3f} ms/iter  (Python→Metal bridge cost for 2 dispatches)")

    # ---- 1. Production Metal: pre-loaded weights (production path) -----------
    print("\n[1] Production Metal — pre-loaded weights (LoadedExpert.forward)")
    print("    (all buffers pre-allocated; only x_view[:]=x per call)")
    expert = tm.load_expert(
        gate_packed, gate_scales, up_packed, up_scales,
        down_packed, down_scales, D_REAL, I_REAL,
    )
    for _ in range(WARMUP):
        expert.forward(x)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        expert.forward(x)
    t1 = time.perf_counter()
    ms_metal_pre = (t1 - t0) / ITERS * 1000.0
    ms_gpu_only  = max(0.0, ms_metal_pre - ms_noop)   # subtract bridge overhead
    gbps_pre     = bytes_tq10 / (ms_metal_pre * 1e-3) / 1e9
    gbps_gpu     = bytes_tq10 / (max(ms_gpu_only, 0.001) * 1e-3) / 1e9
    print(f"    {ms_metal_pre:.3f} ms/iter | {gbps_pre:.1f} GB/s  (wall-clock incl. dispatch)")
    print(f"    {ms_gpu_only:.3f} ms GPU-only (wall - noop) | {gbps_gpu:.1f} GB/s  (pure GPU)")

    # ---- 2. Production Metal: one-shot (arrays copied each call) -------------
    print("\n[2] Production Metal — one-shot (numpy arrays copied each call)")
    for _ in range(WARMUP):
        tm.fused_expert(
            x, gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales, D_REAL, I_REAL,
        )
    t0 = time.perf_counter()
    for _ in range(ITERS):
        tm.fused_expert(
            x, gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales, D_REAL, I_REAL,
        )
    t1 = time.perf_counter()
    ms_metal_one = (t1 - t0) / ITERS * 1000.0
    gbps_one = bytes_tq10 / (ms_metal_one * 1e-3) / 1e9
    print(f"    {ms_metal_one:.3f} ms/iter | {gbps_one:.1f} GB/s  (bottleneck: buffer copy)")

    # ---- 3. CPU fp16 matmul × 3 (torch) -------------------------------------
    print("\n[3] CPU fp16 matmul × 3  (torch, gate + up + down)")
    x_t    = torch.from_numpy(x)
    gate_t = torch.randn(I_REAL, D_REAL, dtype=torch.float16)
    up_t   = torch.randn(I_REAL, D_REAL, dtype=torch.float16)
    down_t = torch.randn(D_REAL, I_REAL, dtype=torch.float16)
    for _ in range(WARMUP):
        m = F.silu(F.linear(x_t, gate_t)) * F.linear(x_t, up_t)
        _ = F.linear(m, down_t)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        m = F.silu(F.linear(x_t, gate_t)) * F.linear(x_t, up_t)
        _ = F.linear(m, down_t)
    t1 = time.perf_counter()
    ms_cpu = (t1 - t0) / ITERS * 1000.0
    gbps_cpu = bytes_fp16 / (ms_cpu * 1e-3) / 1e9
    print(f"    {ms_cpu:.3f} ms/iter | {gbps_cpu:.1f} GB/s  (fp16 weight bytes read)")

    # ---- 4. MPS fp16 matmul × 3 (Apple GPU via torch) -----------------------
    ms_mps = None
    if torch.backends.mps.is_available():
        print("\n[4] MPS fp16 matmul × 3  (torch on Apple GPU)")
        x_m    = x_t.to("mps")
        gate_m = gate_t.to("mps")
        up_m   = up_t.to("mps")
        down_m = down_t.to("mps")
        torch.mps.synchronize()
        for _ in range(WARMUP):
            m = F.silu(F.linear(x_m, gate_m)) * F.linear(x_m, up_m)
            _ = F.linear(m, down_m)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            m = F.silu(F.linear(x_m, gate_m)) * F.linear(x_m, up_m)
            _ = F.linear(m, down_m)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        ms_mps = (t1 - t0) / ITERS * 1000.0
        gbps_mps = bytes_fp16 / (ms_mps * 1e-3) / 1e9
        print(f"    {ms_mps:.3f} ms/iter | {gbps_mps:.1f} GB/s  (fp16 weight bytes)")
    else:
        print("\n[4] MPS: not available")

    # ---- 5. Prototype naive Metal dequant-only (for regression context) ------
    ms_proto = None
    try:
        import metalcompute as mc
        dev_p = mc.Device()
        proto_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "experiments", "metal_shader",
            "tq1_dequant.metal"
        )
        with open(proto_path) as f:
            proto_src = f.read()
        proto_fn = dev_p.kernel(proto_src).function("dequant_tq1")
        N_p  = I_REAL * D_REAL
        nb_p = (N_p + 4) // 5
        ng_p = (N_p + 31) // 32
        pk_p = rng.integers(0, 243, size=nb_p, dtype=np.uint8)
        sc_p = np.ones(ng_p, dtype=np.float16)
        nw_p = np.array([N_p], dtype=np.uint32)
        ob_p = dev_p.buffer(N_p * 2)
        print("\n[5] Prototype Metal — naive dequant-only (one thread per byte)")
        for _ in range(WARMUP):
            proto_fn(nb_p, pk_p, sc_p, ob_p, nw_p)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            proto_fn(nb_p, pk_p, sc_p, ob_p, nw_p)
        t1 = time.perf_counter()
        ms_proto = (t1 - t0) / ITERS * 1000.0
        gbps_proto = nb_p / (ms_proto * 1e-3) / 1e9
        print(f"    {ms_proto:.3f} ms/iter | {gbps_proto:.1f} GB/s  (dequant input bytes)")
    except Exception as e:
        print(f"\n[5] Prototype Metal: SKIP ({e})")

    # ---- Summary -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"  [0] metalcompute dispatch ovhd : {ms_noop:>7.3f} ms  (2 empty dispatches)")
    print(f"  [1] TQ1_0 Metal — pre-loaded  : {ms_metal_pre:>7.3f} ms  |  {gbps_pre:>6.1f} GB/s (wall)")
    print(f"      GPU compute only (wall-noop): {ms_gpu_only:>6.3f} ms  |  {gbps_gpu:>6.1f} GB/s")
    print(f"  [2] TQ1_0 Metal — one-shot    : {ms_metal_one:>7.3f} ms  |  {gbps_one:>6.1f} GB/s")
    print(f"  [3] fp16 CPU torch × 3        : {ms_cpu:>7.3f} ms  |  {gbps_cpu:>6.1f} GB/s")
    if ms_mps is not None:
        print(f"  [4] fp16 MPS torch × 3        : {ms_mps:>7.3f} ms  |  {gbps_mps:>6.1f} GB/s")
    if ms_proto is not None:
        print(f"  [5] Prototype dequant-only    : {ms_proto:>7.3f} ms  |  {gbps_proto:>6.1f} GB/s")

    speedup_cpu = ms_cpu / ms_metal_pre
    speedup_gpu = ms_cpu / max(ms_gpu_only, 0.001)
    print(f"\n  Speedup vs CPU (wall-clock)   : {speedup_cpu:>5.2f}x")
    print(f"  Speedup vs CPU (GPU-only)     : {speedup_gpu:>5.2f}x  (dispatch overhead removed)")
    if ms_mps is not None:
        print(f"  Speedup vs MPS (wall-clock)   : {ms_mps / ms_metal_pre:>5.2f}x")
    if ms_proto is not None:
        print(f"  Speedup vs prototype (wall)   : {ms_proto / ms_metal_pre:>5.2f}x")

    compression = bytes_fp16 / bytes_tq10
    print(f"\n  Weight bytes: fp16={bytes_fp16/1e6:.1f} MB  "
          f"TQ1_0={bytes_tq10/1e6:.1f} MB  ({compression:.1f}× compressed)")
    print(f"  GPU bandwidth (wall)          : {gbps_pre:.1f} GB/s")
    print(f"  GPU bandwidth (compute-only)  : {gbps_gpu:.1f} GB/s  "
          f"(M1 Ultra peak: 800 GB/s)")

    # Tokens/s: 2 experts per token for top-2 routing
    toks_wall = 1000.0 / (ms_metal_pre * 2)
    toks_gpu  = 1000.0 / (max(ms_gpu_only, 0.001) * 2)
    print(f"\n  Est. tokens/s (2 experts, wall)  : ~{toks_wall:.0f} tok/s")
    print(f"  Est. tokens/s (2 experts, GPU)   : ~{toks_gpu:.0f} tok/s")
    print(f"  (wall limited by metalcompute dispatch; GPU-only is the kernel's potential)")
    print()


if __name__ == "__main__":
    if metal_available():
        print("Running correctness smoke-tests...")
        tm = TernaryMetal()
        rng = np.random.default_rng(42)

        M, D = 128, 64
        W = rng.integers(-1, 2, size=(M, D), dtype=np.int8)
        x = rng.standard_normal(D).astype(np.float16)
        packed, scales = pack_matrix_tq1(W)
        ref = ref_tq1_gemv(x, W, scales.reshape(M, -1))
        got = tm.gemv(x, packed, scales, M, D)
        err = float(np.abs(got.astype(np.float32) - ref.astype(np.float32)).max())
        print(f"  tq1_gemv [{M}×{D}]: max_err={err:.6f} {'PASS' if err < 0.01 else 'FAIL'}")

        D2, I2 = 64, 256
        gW = rng.integers(-1, 2, size=(I2, D2), dtype=np.int8)
        uW = rng.integers(-1, 2, size=(I2, D2), dtype=np.int8)
        dW = rng.integers(-1, 2, size=(D2, I2), dtype=np.int8)
        x2 = rng.standard_normal(D2).astype(np.float16)
        gp, gs = pack_matrix_tq1(gW)
        up2, us = pack_matrix_tq1(uW)
        dp, ds = pack_matrix_tq1(dW)
        ref2 = ref_fused_expert(x2, gW, uW, dW)
        got2 = tm.fused_expert(x2, gp, gs, up2, us, dp, ds, D2, I2)
        err2 = float(np.abs(got2.astype(np.float32) - ref2.astype(np.float32)).max())
        print(f"  fused_expert [D={D2},I={I2}]: max_err={err2:.4f} {'PASS' if err2 < 2.0 else 'FAIL'}")

        # Verify pre-loaded path matches one-shot
        expert = tm.load_expert(gp, gs, up2, us, dp, ds, D2, I2)
        pre = expert.forward(x2)
        diff = float(np.abs(pre.astype(np.float32) - got2.astype(np.float32)).max())
        print(f"  preloaded vs one-shot:       max_err={diff:.6f} {'PASS' if diff < 1e-4 else 'FAIL'}")
        print()

    run_benchmark()
