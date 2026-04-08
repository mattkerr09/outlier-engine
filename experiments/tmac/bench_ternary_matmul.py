#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.ternary_matmul import (
    make_ternary_masks,
    pack_bool_mask,
    packed_ternary_linear,
    ternary_linear,
)


def sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bench(name: str, fn, *, iters: int, device: str) -> float:
    for _ in range(10):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    elapsed = time.perf_counter() - t0
    ms = elapsed * 1000.0 / iters
    print(f"{name}_ms={ms:.3f}")
    return ms


def main() -> int:
    device = default_device()
    d_model = 4096
    intermediate = 11008
    iters = 100
    weight = torch.randint(-1, 2, (intermediate, d_model), dtype=torch.int8)
    scale = torch.rand(intermediate, 1, dtype=torch.float16) + 0.1
    x = torch.randn(1, d_model, dtype=torch.float16)

    x_dev = x.to(device)
    weight_float = (weight.to(device=device, dtype=torch.float16) * scale.to(device)).contiguous()
    pos_mask, neg_mask, row_scale = make_ternary_masks(weight.to(device), scale.to(device))
    packed_pos = pack_bool_mask(weight == 1)
    packed_neg = pack_bool_mask(weight == -1)

    print(f"device={device}")
    print(f"shape_x={tuple(x.shape)}")
    print(f"shape_w={tuple(weight.shape)}")
    print(f"iters={iters}")

    standard_ms = bench("standard_float_linear", lambda: F.linear(x_dev, weight_float), iters=iters, device=device)
    mask_ms = bench("ternary_mask_linear", lambda: ternary_linear(x_dev, pos_mask, neg_mask, row_scale), iters=iters, device=device)
    packed_ms = bench(
        "packed_mask_linear",
        lambda: packed_ternary_linear(x_dev, packed_pos, packed_neg, row_scale, device=device, dtype=torch.float16),
        iters=iters,
        device=device,
    )

    best = min(
        [("standard_float_linear", standard_ms), ("ternary_mask_linear", mask_ms), ("packed_mask_linear", packed_ms)],
        key=lambda item: item[1],
    )
    print(f"best={best[0]}")
    print(f"best_ms={best[1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
