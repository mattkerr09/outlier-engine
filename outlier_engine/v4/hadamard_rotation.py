"""Walsh-Hadamard pre-rotation for ternary-quantized MoE experts.

Applies an orthogonal Hadamard rotation H to expert weights before ternary
quantization.  Because H @ H^T = I (scaled), the rotation cancels at inference
when applied symmetrically to both the weight matrix and the input activations:

    y = W @ x
      = (H^T @ H @ W) @ (H^T @ H @ x)     # insert identity twice
      = (H^T) @ (H @ W) @ (H^T) @ (H @ x)  # regroup — but simpler:
      = W_rot @ x_rot                        # rotate weight offline, rotate input online

The key insight: H @ W has a more uniform distribution of values than W alone,
so ternary quantization ({-1, 0, +1}) of H @ W incurs less error than
quantizing W directly.  This is the core idea from QuIP/QuIP# (Chee et al. 2024).

Reference: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
and Lattice Codebooks" — Tseng et al., arXiv:2402.04396
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def build_hadamard(n: int, *, normalize: bool = True) -> torch.Tensor:
    """Construct an n×n Walsh-Hadamard matrix via Sylvester's recursive method.

    Requires n to be a power of 2.  Returns a float32 tensor.
    If normalize=True (default), scales by 1/sqrt(n) so H @ H^T = I.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")

    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    if normalize:
        H = H / math.sqrt(n)
    return H


def _pad_to_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


def fast_hadamard_transform(x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Apply the Walsh-Hadamard transform along the last dimension in O(n log n).

    Uses the butterfly factorization — no explicit matrix construction needed.
    Input shape: (..., n) where n is a power of 2.
    """
    n = x.shape[-1]
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"last dim must be a power of 2, got {n}")

    h = x.float()
    stride = 1
    while stride < n:
        # Butterfly: for each pair of elements separated by `stride`,
        # compute (a+b, a-b).
        idx_even = torch.arange(0, n, 2 * stride, device=x.device)
        for offset in range(stride):
            i0 = idx_even + offset
            i1 = i0 + stride
            a = h[..., i0]
            b = h[..., i1]
            h = h.clone()
            h[..., i0] = a + b
            h[..., i1] = a - b
        stride *= 2

    if normalize:
        h = h / math.sqrt(n)
    return h.to(x.dtype)


def rotate_weight(weight: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to a weight matrix: W_rot = W @ H^T.

    For a linear layer y = W @ x, after rotation:
        y = W_rot @ x_rot = (W @ H^T) @ (H @ x) = W @ (H^T @ H) @ x = W @ x
    The rotation cancels, preserving the original computation.

    Args:
        weight: shape (out_features, in_features)
        H: shape (in_features, in_features), normalized Hadamard matrix
    Returns:
        W_rot: shape (out_features, in_features)
    """
    return (weight.float() @ H.T).to(weight.dtype)


def rotate_input(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Rotate input activations: x_rot = H @ x (applied per-token).

    Args:
        x: shape (..., in_features)
        H: shape (in_features, in_features)
    Returns:
        x_rot: shape (..., in_features)
    """
    return (x.float() @ H.T).to(x.dtype)


def quantize_ternary(w: torch.Tensor, *, threshold: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ternary {-1, 0, +1} with a per-tensor scale.

    Uses the standard threshold-based approach:
        scale = mean(|w|) for non-zero entries
        q = sign(w) * (|w| > threshold * scale)

    Returns (ternary_weights, scale) where ternary_weights ∈ {-1, 0, 1}.
    """
    abs_w = w.abs().float()
    scale = abs_w.mean()
    if threshold is None:
        threshold = 0.7  # standard ternary quantization threshold
    mask = abs_w > (threshold * scale)
    ternary = torch.sign(w) * mask.float()
    return ternary.to(w.dtype), scale


def quantization_error(original: torch.Tensor, quantized: torch.Tensor, scale: torch.Tensor) -> float:
    """Compute normalized L2 quantization error: ||W - s*Q||_2 / ||W||_2."""
    reconstructed = quantized.float() * scale.float()
    error = (original.float() - reconstructed).norm()
    baseline = original.float().norm()
    return (error / baseline).item() if baseline > 0 else float('inf')
