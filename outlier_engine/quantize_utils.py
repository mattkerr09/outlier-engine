"""
quantize_utils.py — INT8 symmetric quantization for shared expert FFN.
OUTLIER-RUNTIME-003

The shared expert in each MoE layer is FP16 (~10.5 GB total across 28 layers).
Quantizing to INT8 with a per-tensor scale cuts this to ~5.3 GB with minimal
quality impact (shared expert weights have low dynamic range).

Quantization scheme:
    scale     = max(|W_i|) / 127     (per output channel)
    W_int8    = round(W / scale).clamp(-127, 127)
    W_approx  = W_int8 * scale

Max absolute error per element <= 0.5 * scale.
"""

from __future__ import annotations

import torch

# Columns per dequant chunk in dequant_int8_matmul.
# Lower = less peak memory; higher = potentially faster on GPU.
QUANT_CHUNK: int = 2048


def _matmul_compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float32 if x.device.type == "cpu" else torch.float16


def quantize_to_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-output-channel symmetric INT8 quantization.

    scale     = max(|weight_i|) / 127
    weight_int8 = round(weight / scale).clamp(-127, 127)

    Input:  any floating-point tensor
    Output: (weight_int8, scale)
              weight_int8 — same shape as input, dtype=int8
              scale       — [out_features, 1] float16 for matrices,
                            scalar float16 for 1D tensors
    """
    weight_f = weight.float()
    if weight_f.ndim <= 1:
        amax = weight_f.abs().max()
    else:
        amax = weight_f.abs().amax(dim=-1, keepdim=True)
    # Clamp denominator to avoid div-by-zero on all-zero weights
    scale = (amax / 127.0).clamp(min=1e-8)
    weight_int8 = (
        weight_f.div(scale)
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return weight_int8, scale.half()


def dequant_int8_matmul(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Compute  y = x @ W.T  where W = weight_int8 * scale.

    Chunked along the in_features dimension to avoid materialising the full
    float weight matrix (saves peak memory during shared expert forward).

    x:           (batch, in_features)    any float
    weight_int8: (out_features, in_features)   int8
    scale:       scalar or [out_features, 1] float

    Returns: (batch, out_features) in the active compute dtype
    """
    out_features, in_features = weight_int8.shape
    batch = x.shape[0]
    compute_dtype = _matmul_compute_dtype(x)

    result = torch.zeros(batch, out_features, device=x.device, dtype=compute_dtype)
    x_f    = x.to(compute_dtype)
    s      = scale.to(compute_dtype)

    for start in range(0, in_features, QUANT_CHUNK):
        end     = min(start + QUANT_CHUNK, in_features)
        # Dequantise this column chunk on the fly — never stores full fp16/fp32 W
        w_chunk = weight_int8[:, start:end].to(compute_dtype) * s   # (out_features, C)
        result += x_f[:, start:end] @ w_chunk.T
        del w_chunk

    return result
