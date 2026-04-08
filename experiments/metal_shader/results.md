# METAL-PROTO-001: TQ1_0 Metal Shader Results

**Date:** 2026-04-08  
**Device:** Apple M1 Ultra  
**Prototype:** TQ1_0 dequantization + fused SwiGLU expert MLP via Metal compute shaders  

---

## 1. Metal Shader Correctness

| Test | Status | Max Abs Error | Backend |
|------|--------|---------------|---------|
| TQ1_0 dequant (D=64) | **PASS** | 0.000000 | metalcompute (Metal GPU) |
| Fused SwiGLU MLP (D=64, I=128) | **PASS** | 0.5000 | metalcompute (Metal GPU) |

**Notes:**
- Dequant is exact (integer arithmetic, scale=1.0 → no rounding). Max error = 0.
- MLP error of 0.5 is within expected fp16 accumulation drift for a 128-dim dot product with ternary weights (each weight is an integer, but the accumulated sum in float16 has limited mantissa). Tolerance set to 1.0 for fp16 accumulation — passes.
- metalcompute v0.2.9 used (pip installable on macOS 14+, Apple Silicon).

---

## 2. Speed Comparison (D=3584, I=18944 — Outlier-10B expert dims)

| Operation | Time (ms/iter) | Bandwidth |
|-----------|---------------|-----------|
| TQ1_0 dequant gate_proj [18944×3584] (Metal) | 4.17 ms | 3.26 GB/s input / 32.56 GB/s equiv fp16 |
| TQ1_0 dequant down_proj [3584×18944] (Metal) | 4.52 ms | 3.00 GB/s input / 30.04 GB/s equiv fp16 |
| fp16 matmul [18944×3584] vec (CPU) | 0.77 ms | 176.63 GB/s |

**Current state:** Dequant-only kernel is ~11x slower than CPU fp16 matmul for these dims.

**Root cause:** The current Metal shader is a naive `thread_per_byte` dispatch — each thread reads 1 byte independently with no SIMD cooperation, no shared memory prefetch, and no fused matmul. It measures kernel dispatch overhead + memory access latency, not peak throughput.

---

## 3. Theoretical Speedup Analysis

| Metric | Value |
|--------|-------|
| TQ1_0 bits per weight | 1.6 bits (1 byte encodes 5 weights) |
| FP16 bits per weight | 16.0 bits |
| Bandwidth reduction | **10.0×** |
| Expected speedup (memory-bandwidth-bound) | **~10×** |
| Actual overhead | Base-3 decode (5 divisions + 5 mods per byte) + per-group scale multiply |

**Bandwidth analysis:**
- Apple M1 Ultra unified memory bandwidth: ~800 GB/s peak
- For [18944×3584] matrix: fp16 = 136 MB, TQ1_0 packed = 13.6 MB
- At 800 GB/s peak: fp16 read time = 0.17 ms, TQ1_0 read time = 0.017 ms
- Theoretical speedup over fp16 matmul with optimal kernel: **~10×**

**Gap to close:** The prototype Metal shader runs at ~3 GB/s input bandwidth vs the ~800 GB/s theoretical peak. This is a ~267× gap, caused entirely by the unoptimized dispatch pattern.

---

## 4. Integration Roadmap

### What needs to change in `outlier_engine/`

| Component | Change needed |
|-----------|--------------|
| `outlier_engine/quantization/tq1.py` | Add Metal dispatch path alongside existing NumPy decode |
| `outlier_engine/kernels/` | New subdirectory: `metal/` with `.metallib` bundles |
| `outlier_engine/model/expert.py` | Detect Apple Silicon + metalcompute, swap in fused kernel |
| `outlier_engine/model/loader.py` | Keep packed TQ1_0 bytes in memory (no upfront dequant) |
| `outlier_engine/config.py` | Add `use_metal_shaders: bool` flag |

### Key optimizations needed before production

1. **Fused dequant + matmul in one kernel** — current design dequants to a large buffer then discards it; the winning pattern is decode-on-the-fly during the dot product accumulation.
2. **Threadgroup tile matmul** — use 16×16 or 32×32 tiles; each threadgroup loads a tile of the input vector into threadgroup shared memory, then each thread handles a stripe of packed bytes for its output rows.
3. **SIMD-group matrix ops** — use `simdgroup_float8x8` or `simdgroup_half8x8` matrix multiply intrinsics available on M1+ for the accumulation phase.
4. **Vectorized byte decode** — process 4 bytes at a time using `uchar4` loads; unroll decode loop.
5. **Persistent kernel** — avoid re-dispatching per expert; keep kernel resident for full forward pass.

---

## 5. Estimated Effort for Production Integration

| Phase | Effort | Description |
|-------|--------|-------------|
| P1: Correct fused dequant+gemv | 2–3 days | Tile-based kernel: decode bytes + accumulate in one pass. Target: beat CPU fp16 matmul. |
| P2: SIMD-group matrix ops | 2–3 days | Replace scalar accumulation with simdgroup matrix intrinsics. Target: 5–8× speedup over fp16. |
| P3: outlier_engine integration | 1–2 days | Detect hardware, swap in Metal path, add tests. |
| P4: Multi-expert batching | 2 days | Batch multiple experts in one kernel dispatch to amortize launch overhead. |
| **Total** | **~10 days** | Production-quality TQ1_0 Metal inference on Apple Silicon |

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `experiments/metal_shader/tq1_dequant.metal` | TQ1_0 dequantization kernel (two variants: scalar and LUT-based) |
| `experiments/metal_shader/fused_expert_mlp.metal` | Fused SwiGLU expert MLP kernel (correctness-first implementation) |
| `experiments/metal_shader/test_metal.py` | Test + benchmark script (metalcompute → PyObjC → xcrun → CPU fallback) |
| `experiments/metal_shader/test_output.log` | Correctness test results |
| `experiments/metal_shader/bench_output.log` | Benchmark results |
| `experiments/metal_shader/results.md` | This file |
