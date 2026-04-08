#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim: int = 3584, intermediate_dim: int = 18944) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _bench(module: nn.Module, x: torch.Tensor, *, device: str, label: str, steps: int = 100) -> float:
    with torch.no_grad():
        for _ in range(10):
            _ = module(x)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = module(x)
        _sync(device)
    ms = (time.perf_counter() - t0) * 1000.0 / steps
    print(f"{label}_ms={ms:.3f}")
    return ms


def main() -> int:
    device = _default_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    x = torch.randn(1, 3584, device=device, dtype=dtype)

    eager = SimpleMLP().to(device=device, dtype=dtype).eval()
    print(f"device={device}")
    print(f"dtype={dtype}")
    print("module=SimpleMLP(gate+up+silu+down)")
    eager_ms = _bench(eager, x, device=device, label="eager")

    compile_error = None
    compiled_ms = None
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(eager, mode="reduce-overhead")
            compiled_ms = _bench(compiled, x, device=device, label="torch_compile")
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            compile_error = exc
            print(f"torch_compile_error={type(exc).__name__}: {exc}")

    trace_error = None
    traced_ms = None
    try:
        traced = torch.jit.trace(eager, x)
        traced_ms = _bench(traced, x, device=device, label="torch_jit_trace")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        trace_error = exc
        print(f"torch_jit_trace_error={type(exc).__name__}: {exc}")

    if compiled_ms is not None:
        print(f"compile_speedup_vs_eager={eager_ms / compiled_ms:.3f}")
    if traced_ms is not None:
        print(f"trace_speedup_vs_eager={eager_ms / traced_ms:.3f}")
    if compile_error is not None and trace_error is not None:
        print("result=neither_compile_nor_trace_worked")
    elif compiled_ms is not None or traced_ms is not None:
        print("result=at_least_one_compilation_strategy_worked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
