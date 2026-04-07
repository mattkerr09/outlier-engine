from __future__ import annotations

import numpy as np


def decode_packed_ternary(packed: np.ndarray, original_in_features: int) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    codes = np.stack([b0, b1, b2, b3], axis=-1).reshape(packed.shape[0], -1)
    out = np.zeros_like(codes, dtype=np.int8)
    out[codes == 1] = 1
    out[codes == 3] = -1
    return out[:, :original_in_features]


def ternary_matmul_numpy(
    x: np.ndarray,
    packed_weight: np.ndarray,
    scale: np.ndarray,
    original_in_features: int,
) -> np.ndarray:
    weight = decode_packed_ternary(packed_weight, original_in_features).astype(np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if scale.ndim == 0:
        scaled = weight * scale
    else:
        scaled = weight * scale.reshape(-1, 1)
    return np.asarray(x, dtype=np.float32) @ scaled.T
