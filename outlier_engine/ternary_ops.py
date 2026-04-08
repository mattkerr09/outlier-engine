"""
ternary_ops.py — 2-bit ternary weight packing and memory-efficient matmul kernels.
OUTLIER-RUNTIME-003

Key insight: ternary weights are {-1, 0, +1} as int8, so only 2 bits are needed
per weight.  Packing 4 weights per byte gives 4x RAM reduction for expert storage.

Encoding (LSB-first within byte):
    0b00 = 0   (zero weight)
    0b01 = +1  (positive weight)
    0b11 = -1  (negative weight)
    0b10 = 0   (unused; decoded gracefully as zero)

The encoding trick:
    int8(-1) = 0xFF;  0xFF & 0x03 = 3 = 0b11  ✓
    int8( 0) = 0x00;  0x00 & 0x03 = 0 = 0b00  ✓
    int8(+1) = 0x01;  0x01 & 0x03 = 1 = 0b01  ✓

So  `code = weights_int8.to(uint8) & 0x03`  encodes in one op.

Matmul strategy (sign decomposition, no custom CUDA kernels):
    y = x @ W.T * scale
      = x @ (W_pos - W_neg).T * scale
      = (x @ W_pos.T - x @ W_neg.T) * scale

    Chunked to avoid materialising the full float matrix:
    process CHUNK_COLS columns at a time, accumulate into a float32 result.
"""

from __future__ import annotations

import torch

# Number of *packed* columns per matmul chunk (= CHUNK_COLS * 4 original columns).
# Lower values use less peak memory; higher values may be faster on GPU.
CHUNK_COLS: int = 256   # 256 packed cols = 1024 original cols


def _matmul_compute_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float32 if x.device.type == "cpu" else torch.float16


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def ternarize_per_channel(
    weight_fp16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize an FP16 weight matrix to ternary {-1, 0, +1} with per-output-row
    scaling.  Compared to per-tensor scaling, this preserves outlier channels
    that a single global scale would clip or zero out.

    scale[i] = mean(|W[i, :]|)   — one scale per output row
    ternary[i, j] = clamp(round(W[i,j] / scale[i]), -1, 1)

    Scale shape: (out_features, 1) — keepdim so it broadcasts directly against
    the weight shape (out_features, in_features) without unsqueeze at call sites:
        W_approx = ternary.float() * scale  →  (out, in) * (out, 1) = (out, in)

    The (out, 1) shape also works transparently in ternary_matmul_direct /
    ternary_matmul_packed via their existing `s.view(1, -1)` path:
        s.view(1, -1) on (out, 1) → (1, out) → result*(1, out) per-channel.

    Args:
        weight_fp16: (out_features, in_features)  any floating-point dtype

    Returns:
        ternary: (out_features, in_features)  int8, values in {-1, 0, +1}
        scale:   (out_features, 1)            float16
    """
    w = weight_fp16.float()
    scale   = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)  # (out, 1)
    ternary = (w / scale).round().clamp(-1, 1).to(torch.int8)
    return ternary, scale.to(torch.float16)


# ---------------------------------------------------------------------------
# Packing / unpacking
# ---------------------------------------------------------------------------

def pack_ternary_2bit(weights_int8: torch.Tensor) -> torch.Tensor:
    """
    Pack int8 ternary weights {-1, 0, +1} → uint8 with 4 weights per byte.

    Input:  (out_features, in_features)  int8, values in {-1, 0, +1}
    Output: (out_features, ceil(in_features / 4))  uint8

    Padding: if in_features is not a multiple of 4, zero-pad (0b00 = 0).
    """
    out_features, in_features = weights_int8.shape

    # Pad in_features to a multiple of 4 using zero weights (0b00)
    pad = (4 - in_features % 4) % 4
    if pad > 0:
        weights_int8 = torch.cat(
            [
                weights_int8,
                torch.zeros(out_features, pad, dtype=torch.int8,
                             device=weights_int8.device),
            ],
            dim=1,
        )
    padded_in = in_features + pad  # multiple of 4

    # Encode to 2-bit codes via the mask trick (see module docstring)
    # Result: uint8 in {0, 1, 3}
    code = weights_int8.to(torch.uint8) & 0x03   # (out_features, padded_in)

    # Reshape to groups of 4, then pack into bytes (LSB-first)
    code = code.view(out_features, padded_in // 4, 4)   # (out, padded_in/4, 4)

    # Use int32 intermediates to avoid uint8 overflow on some backends
    packed = (
          code[:, :, 0].to(torch.int32)
        | (code[:, :, 1].to(torch.int32) << 2)
        | (code[:, :, 2].to(torch.int32) << 4)
        | (code[:, :, 3].to(torch.int32) << 6)
    ).to(torch.uint8)

    return packed   # (out_features, padded_in // 4)


def unpack_ternary_2bit(
    packed: torch.Tensor,
    original_in_features: int,
) -> torch.Tensor:
    """
    Reverse of pack_ternary_2bit.  Used for correctness verification.

    Decode: 0b00→0, 0b01→+1, 0b11→-1, 0b10→0 (unused → 0)

    Input:  (out_features, packed_cols)  uint8
    Output: (out_features, original_in_features)  int8
    """
    out_features = packed.shape[0]

    # Extract four 2-bit groups per byte
    b0 = (packed & 0x03)                   # (out_features, packed_cols)
    b1 = ((packed >> 2) & 0x03)
    b2 = ((packed >> 4) & 0x03)
    b3 = ((packed >> 6) & 0x03)

    # Interleave into (out_features, packed_cols * 4)
    codes = torch.stack([b0, b1, b2, b3], dim=2).reshape(out_features, -1)

    # Decode codes to {-1, 0, +1}
    result = torch.zeros_like(codes, dtype=torch.int8)
    result[codes == 1] = 1
    result[codes == 3] = -1
    # code 2 → 0 (graceful no-op)

    return result[:, :original_in_features]


# ---------------------------------------------------------------------------
# Memory-efficient matmul on packed 2-bit weights
# ---------------------------------------------------------------------------

def ternary_matmul_packed(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_in_features: int,
) -> torch.Tensor:
    """
    Compute  y = (x @ W.T) * scale  where W is 2-bit packed ternary.

    Never materialises the full float weight matrix — processes in column
    chunks of size CHUNK_COLS packed columns (= CHUNK_COLS*4 original cols).

    x:              (batch, in_features)   float16 or float32
    packed_weight:  (out_features, ceil(in_features/4))  uint8
    scale:          scalar  OR  (out_features,)  float  (any fp dtype)
    original_in_features:  actual in_features before 4-alignment padding

    Returns: (batch, out_features)  float32
    """
    out_features  = packed_weight.shape[0]
    packed_cols   = packed_weight.shape[1]
    batch         = x.shape[0]
    padded_in     = packed_cols * 4

    compute_dtype = _matmul_compute_dtype(x)
    result = torch.zeros(batch, out_features, device=x.device, dtype=compute_dtype)
    x_f    = x.to(compute_dtype)

    # Pad x to match the padded weight dimension (zero-pad)
    if original_in_features < padded_in:
        x_f = torch.cat(
            [
                x_f,
                torch.zeros(batch, padded_in - original_in_features,
                             device=x.device, dtype=compute_dtype),
            ],
            dim=1,
        )

    for start in range(0, packed_cols, CHUNK_COLS):
        end   = min(start + CHUNK_COLS, packed_cols)
        chunk = packed_weight[:, start:end]   # (out_features, C)  uint8

        # Unpack 2-bit codes in this chunk
        b0 = (chunk & 0x03)
        b1 = ((chunk >> 2) & 0x03)
        b2 = ((chunk >> 4) & 0x03)
        b3 = ((chunk >> 6) & 0x03)
        w_codes = torch.stack([b0, b1, b2, b3], dim=2).reshape(out_features, -1)
        # w_codes: (out_features, C*4)  uint8, values in {0, 1, 3}

        # Sign decomposition (1 byte per weight, not 4)
        w_pos = (w_codes == 1).to(compute_dtype)   # +1 positions  (out_features, C*4)
        w_neg = (w_codes == 3).to(compute_dtype)   # -1 positions

        x_chunk = x_f[:, start * 4 : end * 4]   # (batch, C*4)
        result += x_chunk @ w_pos.T
        result -= x_chunk @ w_neg.T

        del w_pos, w_neg, w_codes   # free chunk memory

    # Apply scale
    s = scale.to(compute_dtype)
    if s.dim() == 0:
        result = result * s
    else:
        result = result * s.view(1, -1)

    return result.float()


# ---------------------------------------------------------------------------
# Memory-efficient matmul on int8 weights (fallback / device-side compute)
# ---------------------------------------------------------------------------

def ternary_matmul_direct(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Compute  y = (x @ W.T) * scale  where W is int8 ternary {-1, 0, +1}.

    Chunked to avoid materialising the full float weight copy.

    x:           (batch, in_features)   float16 or float32
    weight_int8: (out_features, in_features)   int8
    scale:       scalar  OR  (out_features,)   float

    Returns: (batch, out_features)  float32
    """
    _, in_features = weight_int8.shape
    batch          = x.shape[0]
    out_features   = weight_int8.shape[0]
    chunk_size     = CHUNK_COLS * 4   # 1024 original cols per chunk

    compute_dtype = _matmul_compute_dtype(x)
    result = torch.zeros(batch, out_features, device=x.device, dtype=compute_dtype)
    x_f    = x.to(compute_dtype)

    for start in range(0, in_features, chunk_size):
        end     = min(start + chunk_size, in_features)
        w_chunk = weight_int8[:, start:end]    # (out_features, C)  int8
        x_chunk = x_f[:, start:end]            # (batch, C)

        # Sign decomposition: avoids any float copy of the full weight
        w_pos = (w_chunk == 1).to(compute_dtype)
        w_neg = (w_chunk == -1).to(compute_dtype)

        result += x_chunk @ w_pos.T
        result -= x_chunk @ w_neg.T

        del w_pos, w_neg

    s = scale.to(compute_dtype)
    if s.dim() == 0:
        result = result * s
    else:
        result = result * s.view(1, -1)

    return result.float()
