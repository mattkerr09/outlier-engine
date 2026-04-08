#!/usr/bin/env python3
"""
OUTLIER-ENGINE-BATCHED-GEMM-001: Task 4 — Profile gather vs compute time.

Measures how long torch.stack (the gather) takes vs the batched matmuls.

Usage:
    cd ~/outlier-engine && source .venv/bin/activate
    python experiments/batched_gemm/profile_gather.py 2>&1 | tee experiments/batched_gemm/gather_profile.log
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F


def _sync(device):
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.synchronize()


def profile_gather(n_trials=100):
    # Replicate Outlier-10B expert dimensions (approx)
    hidden_dim = 3584        # Qwen2.5-7B hidden_dim
    intermediate_dim = 18944 # Qwen2.5-7B intermediate_dim (approx for 10B)
    n_experts = 2
    dtype = torch.float16

    device_name = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    device = torch.device(device_name)
    print(f"device={device}")
    print(f"hidden_dim={hidden_dim}  intermediate_dim={intermediate_dim}  n_experts={n_experts}  dtype={dtype}")

    # Create synthetic expert weight tensors (dequantized, float16)
    gate_ws = [torch.randn(intermediate_dim, hidden_dim, dtype=dtype, device=device) for _ in range(n_experts)]
    up_ws   = [torch.randn(intermediate_dim, hidden_dim, dtype=dtype, device=device) for _ in range(n_experts)]
    down_ws = [torch.randn(hidden_dim, intermediate_dim, dtype=dtype, device=device) for _ in range(n_experts)]
    x = torch.randn(1, hidden_dim, dtype=dtype, device=device)
    x_col = x.squeeze(0).unsqueeze(-1)  # [H, 1]

    # Warmup
    for _ in range(5):
        gate_w = torch.stack(gate_ws, dim=0)
        up_w   = torch.stack(up_ws, dim=0)
        down_w = torch.stack(down_ws, dim=0)
        gate_out = torch.matmul(gate_w, x_col).squeeze(-1)
        up_out   = torch.matmul(up_w, x_col).squeeze(-1)
        act_out  = F.silu(gate_out) * up_out
        down_out = torch.matmul(down_w, act_out.unsqueeze(-1)).squeeze(-1)
    _sync(device)

    # ---- Measure gather (torch.stack) ----
    gather_times = []
    for _ in range(n_trials):
        _sync(device)
        t0 = time.perf_counter()
        gate_w = torch.stack(gate_ws, dim=0)
        up_w   = torch.stack(up_ws, dim=0)
        down_w = torch.stack(down_ws, dim=0)
        _sync(device)
        gather_times.append((time.perf_counter() - t0) * 1000.0)

    gather_ms_avg = sum(gather_times) / len(gather_times)
    gather_ms_p50 = sorted(gather_times)[len(gather_times) // 2]

    # ---- Measure compute (4 matmuls) ----
    gate_w = torch.stack(gate_ws, dim=0)
    up_w   = torch.stack(up_ws, dim=0)
    down_w = torch.stack(down_ws, dim=0)
    compute_times = []
    for _ in range(n_trials):
        _sync(device)
        t0 = time.perf_counter()
        gate_out = torch.matmul(gate_w, x_col).squeeze(-1)
        up_out   = torch.matmul(up_w, x_col).squeeze(-1)
        act_out  = F.silu(gate_out) * up_out
        down_out = torch.matmul(down_w, act_out.unsqueeze(-1)).squeeze(-1)
        _sync(device)
        compute_times.append((time.perf_counter() - t0) * 1000.0)

    compute_ms_avg = sum(compute_times) / len(compute_times)
    compute_ms_p50 = sorted(compute_times)[len(compute_times) // 2]

    # ---- Measure total (gather + compute) ----
    total_times = []
    for _ in range(n_trials):
        _sync(device)
        t0 = time.perf_counter()
        gate_w = torch.stack(gate_ws, dim=0)
        up_w   = torch.stack(up_ws, dim=0)
        down_w = torch.stack(down_ws, dim=0)
        gate_out = torch.matmul(gate_w, x_col).squeeze(-1)
        up_out   = torch.matmul(up_w, x_col).squeeze(-1)
        act_out  = F.silu(gate_out) * up_out
        down_out = torch.matmul(down_w, act_out.unsqueeze(-1)).squeeze(-1)
        _sync(device)
        total_times.append((time.perf_counter() - t0) * 1000.0)

    total_ms_avg = sum(total_times) / len(total_times)
    total_ms_p50 = sorted(total_times)[len(total_times) // 2]

    gather_pct = gather_ms_avg / max(total_ms_avg, 1e-9) * 100

    print(f"n_trials={n_trials}")
    print(f"gather_ms_avg={gather_ms_avg:.3f}  gather_ms_p50={gather_ms_p50:.3f}")
    print(f"compute_ms_avg={compute_ms_avg:.3f}  compute_ms_p50={compute_ms_p50:.3f}")
    print(f"total_ms_avg={total_ms_avg:.3f}  total_ms_p50={total_ms_p50:.3f}")
    print(f"gather_pct_of_total={gather_pct:.1f}%")


if __name__ == "__main__":
    profile_gather()
