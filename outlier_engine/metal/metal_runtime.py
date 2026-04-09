"""
metal_runtime.py — Production TQ1_0 Metal GEMV wrapper.

Two usage patterns:

1. One-shot (benchmarking/tests — copies arrays each call):
     tm = TernaryMetal()
     out = tm.fused_expert(x, gate_packed, gate_scales, ...)

2. Pre-loaded (production inference — weight arrays copied once to Metal buffers):
     expert = tm.load_expert(gate_packed, gate_scales, up_packed, up_scales,
                             down_packed, down_scales, D, I)
     out = expert.forward(x)  # only x is copied per call (~7 KB)

Pattern 2 is the target for real inference: it eliminates the per-call 13 MB
numpy→Metal copy that causes the one-shot path to run at only 4 GB/s.
"""

from __future__ import annotations

import os
import numpy as np

_SHADER_PATH = os.path.join(os.path.dirname(__file__), "ternary_gemv.metal")

_metal_ok: bool | None = None


def metal_available() -> bool:
    global _metal_ok
    if _metal_ok is not None:
        return _metal_ok
    try:
        import metalcompute as mc  # type: ignore
        mc.Device()
        _metal_ok = True
    except Exception:
        _metal_ok = False
    return _metal_ok


def _load_shader() -> str:
    with open(_SHADER_PATH) as f:
        return f.read()


def _np_to_metal_buf(device, arr: np.ndarray):
    """Copy a numpy array into a persistent Metal shared-memory buffer."""
    raw = arr.tobytes()
    buf = device.buffer(len(raw))
    # metalcompute buffers support the buffer protocol — write via memoryview.
    mv = memoryview(buf).cast("B")
    mv[:] = raw
    return buf


class LoadedExpert:
    """
    A TQ1_0 expert whose weights are pre-uploaded to Metal shared buffers.
    Forward pass only transfers the small input vector x (~7 KB for D=3584).
    """

    def __init__(
        self,
        device,
        fn_gate_up,
        fn_down,
        gate_buf,
        gate_scales_buf,
        up_buf,
        up_scales_buf,
        down_buf,
        down_scales_buf,
        D: int,
        I: int,
    ) -> None:
        self._dev            = device
        self._fn_gate_up     = fn_gate_up
        self._fn_down        = fn_down
        self._gate_buf       = gate_buf
        self._gate_scales    = gate_scales_buf
        self._up_buf         = up_buf
        self._up_scales      = up_scales_buf
        self._down_buf       = down_buf
        self._down_scales    = down_scales_buf
        self._D              = D
        self._I              = I
        self._D_arr          = np.array([D], dtype=np.uint32)
        self._I_arr          = np.array([I], dtype=np.uint32)
        self._mid_buf        = device.buffer(I * 4)   # float32 intermediate
        self._out_buf        = device.buffer(D * 2)   # float16 output
        # Pre-allocate x buffer so forward() is a memoryview write, not a copy.
        self._x_buf          = device.buffer(D * 2)   # float16 input
        self._x_view         = np.frombuffer(self._x_buf, dtype=np.float16)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [D] float16 numpy array.
        Returns [D] float16 numpy array.
        """
        # Write x into pre-allocated Metal shared buffer (avoids per-call copy).
        self._x_view[:] = x
        self._fn_gate_up(
            self._I,
            self._x_buf,
            self._gate_buf,
            self._gate_scales,
            self._up_buf,
            self._up_scales,
            self._mid_buf,
            self._I_arr,
            self._D_arr,
        )
        self._fn_down(
            self._D,
            self._mid_buf,
            self._down_buf,
            self._down_scales,
            self._out_buf,
            self._D_arr,
            self._I_arr,
        )
        return np.frombuffer(self._out_buf, dtype=np.float16).copy()


class TernaryMetal:
    """
    Production Metal compute wrapper for TQ1_0 GEMV and fused SwiGLU expert.
    """

    def __init__(self) -> None:
        if not metal_available():
            raise RuntimeError(
                "Metal not available. Install: pip install metalcompute"
            )
        import metalcompute as mc  # type: ignore
        self._dev = mc.Device()
        src = _load_shader()
        pipeline = self._dev.kernel(src)
        self._fn_gemv    = pipeline.function("tq1_gemv")
        self._fn_gate_up = pipeline.function("tq1_gate_up_swiglu")
        self._fn_down    = pipeline.function("tq1_down_proj")

    # ------------------------------------------------------------------
    # One-shot GEMV (copies arrays per call — useful for testing)
    # ------------------------------------------------------------------
    def gemv(
        self,
        x:        np.ndarray,  # [D] float16
        W_packed: np.ndarray,  # [M * bytes_per_row] uint8
        W_scales: np.ndarray,  # [M * scales_per_row] float16
        M: int,
        D: int,
    ) -> np.ndarray:           # [M] float16
        out_buf = self._dev.buffer(M * 2)
        self._fn_gemv(
            M, x, W_packed, W_scales, out_buf,
            np.array([M], dtype=np.uint32),
            np.array([D], dtype=np.uint32),
        )
        return np.frombuffer(out_buf, dtype=np.float16).copy()

    # ------------------------------------------------------------------
    # Pre-load expert weights into Metal buffers (production path)
    # ------------------------------------------------------------------
    def load_expert(
        self,
        gate_packed:  np.ndarray,  # [I * bytes_gu] uint8
        gate_scales:  np.ndarray,  # [I * scales_gu] float16
        up_packed:    np.ndarray,  # [I * bytes_gu] uint8
        up_scales:    np.ndarray,  # [I * scales_gu] float16
        down_packed:  np.ndarray,  # [D * bytes_dn] uint8
        down_scales:  np.ndarray,  # [D * scales_dn] float16
        D: int,
        I: int,
    ) -> LoadedExpert:
        """Copy weights to Metal shared buffers once; return a LoadedExpert."""
        return LoadedExpert(
            device          = self._dev,
            fn_gate_up      = self._fn_gate_up,
            fn_down         = self._fn_down,
            gate_buf        = _np_to_metal_buf(self._dev, gate_packed),
            gate_scales_buf = _np_to_metal_buf(self._dev, gate_scales),
            up_buf          = _np_to_metal_buf(self._dev, up_packed),
            up_scales_buf   = _np_to_metal_buf(self._dev, up_scales),
            down_buf        = _np_to_metal_buf(self._dev, down_packed),
            down_scales_buf = _np_to_metal_buf(self._dev, down_scales),
            D=D, I=I,
        )

    # ------------------------------------------------------------------
    # One-shot fused expert (copies arrays per call — for testing only)
    # ------------------------------------------------------------------
    def fused_expert(
        self,
        x:            np.ndarray,
        gate_packed:  np.ndarray,
        gate_scales:  np.ndarray,
        up_packed:    np.ndarray,
        up_scales:    np.ndarray,
        down_packed:  np.ndarray,
        down_scales:  np.ndarray,
        D: int,
        I: int,
    ) -> np.ndarray:
        mid_buf = self._dev.buffer(I * 4)
        out_buf = self._dev.buffer(D * 2)
        I_arr   = np.array([I], dtype=np.uint32)
        D_arr   = np.array([D], dtype=np.uint32)
        self._fn_gate_up(
            I, x, gate_packed, gate_scales, up_packed, up_scales,
            mid_buf, I_arr, D_arr,
        )
        self._fn_down(
            D, mid_buf, down_packed, down_scales, out_buf, D_arr, I_arr,
        )
        return np.frombuffer(out_buf, dtype=np.float16).copy()

    # ------------------------------------------------------------------
    # Benchmark helper
    # ------------------------------------------------------------------
    def benchmark(
        self,
        D: int = 3584,
        I: int = 18944,
        warmup: int = 20,
        iters: int = 200,
        seed: int = 0,
    ) -> dict:
        import time
        rng = np.random.default_rng(seed)
        bytes_gu  = (D + 4) // 5
        scales_gu = (D + 31) // 32
        bytes_dn  = (I + 4) // 5
        scales_dn = (I + 31) // 32

        x            = rng.standard_normal(D).astype(np.float16)
        gate_packed  = rng.integers(0, 243, size=I * bytes_gu,  dtype=np.uint8)
        gate_scales  = (rng.standard_normal(I * scales_gu) * 0.01).astype(np.float16)
        up_packed    = rng.integers(0, 243, size=I * bytes_gu,  dtype=np.uint8)
        up_scales    = (rng.standard_normal(I * scales_gu) * 0.01).astype(np.float16)
        down_packed  = rng.integers(0, 243, size=D * bytes_dn,  dtype=np.uint8)
        down_scales  = (rng.standard_normal(D * scales_dn) * 0.01).astype(np.float16)

        # Pre-load weights into Metal buffers
        expert = self.load_expert(
            gate_packed, gate_scales, up_packed, up_scales,
            down_packed, down_scales, D, I,
        )

        for _ in range(warmup):
            expert.forward(x)

        t0 = time.perf_counter()
        for _ in range(iters):
            expert.forward(x)
        t1 = time.perf_counter()

        ms_per = (t1 - t0) / iters * 1000.0
        bytes_read = (
            I * bytes_gu + I * scales_gu * 2 +
            I * bytes_gu + I * scales_gu * 2 +
            D * bytes_dn + D * scales_dn * 2
        )
        gbps = bytes_read / (ms_per * 1e-3) / 1e9
        fp16_bytes = (I * D + I * D + D * I) * 2
        return {
            "ms_per_iter":        ms_per,
            "gbps_tq10":          gbps,
            "fp16_weight_bytes":  fp16_bytes,
            "tq10_weight_bytes":  bytes_read,
            "compression_ratio":  fp16_bytes / bytes_read,
        }
