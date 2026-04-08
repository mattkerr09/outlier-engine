#!/usr/bin/env python3
"""
OUTLIER-ENGINE-BATCHED-GEMM-001: Task 1 — Count kernel launches per token.

Counts how many torch compute operations fire per token:
  sequential path: 5 ops × n_experts (F.linear×3, F.silu, multiply)
  batched path:    5 ops total (2×matmul, silu, mul, matmul)

Also measures gather (torch.stack) overhead.

Usage:
    cd ~/outlier-engine && source .venv/bin/activate
    python experiments/batched_gemm/count_launches.py 2>&1 | tee experiments/batched_gemm/launch_count.log
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from outlier_engine.paging import _ExpertWeights, _run_single_token_experts_batched, _run_expert
from outlier_engine.batched_expert import BatchedExpertMLP


# ---------------------------------------------------------------------------
# Kernel launch counter via op patching
# ---------------------------------------------------------------------------

_op_count = 0
_COUNTING = False


def _patch_and_count():
    """Context manager that counts compute ops."""
    import contextlib

    @contextlib.contextmanager
    def _counter():
        global _op_count, _COUNTING
        _op_count = 0
        _COUNTING = True

        orig_linear = F.linear
        orig_matmul = torch.matmul
        orig_bmm = torch.bmm
        orig_silu = F.silu
        orig_mul = torch.Tensor.mul

        def _c_linear(inp, w, b=None):
            global _op_count
            if _COUNTING:
                _op_count += 1
            return orig_linear(inp, w, b)

        def _c_matmul(a, b, *args, **kwargs):
            global _op_count
            if _COUNTING:
                _op_count += 1
            return orig_matmul(a, b, *args, **kwargs)

        def _c_silu(x, *args, **kwargs):
            global _op_count
            if _COUNTING:
                _op_count += 1
            return orig_silu(x, *args, **kwargs)

        F.linear = _c_linear
        torch.matmul = _c_matmul
        F.silu = _c_silu

        try:
            yield
        finally:
            _COUNTING = False
            F.linear = orig_linear
            torch.matmul = orig_matmul
            torch.bmm = orig_bmm
            F.silu = orig_silu

    return _counter


counter = _patch_and_count()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_expert(hidden_dim: int, intermediate_dim: int, device: torch.device, dequantized: bool) -> _ExpertWeights:
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    return _ExpertWeights(
        gate_w=torch.randn(intermediate_dim, hidden_dim, dtype=dtype, device=device),
        gate_s=torch.ones(1, dtype=dtype, device=device),
        up_w=torch.randn(intermediate_dim, hidden_dim, dtype=dtype, device=device),
        up_s=torch.ones(1, dtype=dtype, device=device),
        down_w=torch.randn(hidden_dim, intermediate_dim, dtype=dtype, device=device),
        down_s=torch.ones(1, dtype=dtype, device=device),
        packed=False,
        dequantized=dequantized,
    )


def main():
    device_name = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    device = torch.device(device_name)
    print(f"device={device_name}")

    hidden_dim = 128
    intermediate_dim = 512
    n_experts = 2
    n_moe_layers = 17  # approx MoE layers in Outlier-10B

    # dequantized=True → F.linear path (fast, no int8 dequant needed)
    experts = [make_expert(hidden_dim, intermediate_dim, device, dequantized=True) for _ in range(n_experts)]
    x = torch.randn(1, hidden_dim, dtype=experts[0].gate_w.dtype, device=device)
    routing_weights = torch.tensor([0.6, 0.4], device=device, dtype=experts[0].gate_w.dtype)

    runner = BatchedExpertMLP()

    # ---- Sequential: per-expert, one at a time ----
    with counter():
        for exp in experts:
            _run_expert(x, exp)
    seq_launches_per_layer = _op_count
    print(f"sequential_launches_per_layer={seq_launches_per_layer}  ({seq_launches_per_layer // n_experts} per expert × {n_experts} experts)")

    # ---- Batched: all experts in one pass ----
    with counter():
        _ = _run_single_token_experts_batched(x, experts)
    batched_launches_per_layer = _op_count
    print(f"batched_launches_per_layer={batched_launches_per_layer}  (all {n_experts} experts batched)")

    # ---- BatchedExpertMLP (new class) ----
    with counter():
        _ = runner.forward(x, experts, routing_weights)
    new_batched_launches = _op_count
    print(f"new_BatchedExpertMLP_launches_per_layer={new_batched_launches}")

    # ---- Extrapolate to full model ----
    seq_total = seq_launches_per_layer * n_moe_layers
    batched_total = batched_launches_per_layer * n_moe_layers
    print()
    print(f"=== Extrapolation to {n_moe_layers} MoE layers, top_k={n_experts} ===")
    print(f"sequential_total_launches_per_token={seq_total}")
    print(f"batched_total_launches_per_token={batched_total}")
    if batched_total > 0:
        print(f"reduction_factor={seq_total / batched_total:.1f}x")
    else:
        print("reduction_factor=N/A (batched_total=0)")

    # ---- Measure gather overhead ----
    n_trials = 50

    def sync():
        if device_name == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # Warmup
    for _ in range(5):
        gate_w = torch.stack([e.gate_w for e in experts], dim=0)
        up_w   = torch.stack([e.up_w   for e in experts], dim=0)
        down_w = torch.stack([e.down_w for e in experts], dim=0)
    sync()

    gather_times = []
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        gate_w = torch.stack([e.gate_w for e in experts], dim=0)
        up_w   = torch.stack([e.up_w   for e in experts], dim=0)
        down_w = torch.stack([e.down_w for e in experts], dim=0)
        sync()
        gather_times.append((time.perf_counter() - t0) * 1000.0)

    compute_times = []
    gate_w = torch.stack([e.gate_w for e in experts], dim=0)
    up_w   = torch.stack([e.up_w   for e in experts], dim=0)
    down_w = torch.stack([e.down_w for e in experts], dim=0)
    x_col  = x.squeeze(0).unsqueeze(-1)
    for _ in range(n_trials):
        sync()
        t0 = time.perf_counter()
        g = torch.matmul(gate_w, x_col).squeeze(-1)
        u = torch.matmul(up_w,   x_col).squeeze(-1)
        a = F.silu(g) * u
        d = torch.matmul(down_w, a.unsqueeze(-1)).squeeze(-1)
        sync()
        compute_times.append((time.perf_counter() - t0) * 1000.0)

    g_avg = sum(gather_times) / len(gather_times)
    c_avg = sum(compute_times) / len(compute_times)
    total = g_avg + c_avg
    print()
    print(f"=== Gather vs compute (n_experts={n_experts}, dim={hidden_dim}x{intermediate_dim}) ===")
    print(f"gather_ms_avg={g_avg:.4f}")
    print(f"compute_ms_avg={c_avg:.4f}")
    print(f"gather_pct_of_total={g_avg / max(total, 1e-9) * 100:.1f}%")


if __name__ == "__main__":
    main()
