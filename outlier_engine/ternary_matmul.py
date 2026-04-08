from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PackedBoolMask:
    packed: torch.Tensor
    shape: Tuple[int, ...]


def make_ternary_masks(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert {-1,0,+1} ternary weights into positive/negative float masks plus row scale."""
    out_dtype = torch.float32 if weight.device.type == "cpu" else torch.float16
    pos = (weight == 1).to(dtype=out_dtype)
    neg = (weight == -1).to(dtype=out_dtype)
    row_scale = scale.reshape(-1).to(dtype=out_dtype)
    return pos.contiguous(), neg.contiguous(), row_scale.contiguous()


def ternary_linear(
    x: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Mask-based ternary linear layer using additions/subtractions plus row scale."""
    compute_dtype = positive_mask.dtype
    x_dev = x if (x.dtype == compute_dtype and x.device == positive_mask.device) else x.to(
        device=positive_mask.device, dtype=compute_dtype
    )
    pos = F.linear(x_dev, positive_mask)
    neg = F.linear(x_dev, negative_mask)
    return (pos - neg) * scale.view(1, -1)


def pack_bool_mask(mask: torch.Tensor) -> PackedBoolMask:
    """Pack a boolean mask into uint8 bytes on CPU using numpy.packbits."""
    mask_np = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    packed = np.packbits(mask_np.reshape(-1), bitorder="little")
    return PackedBoolMask(
        packed=torch.from_numpy(packed.copy()),
        shape=tuple(mask.shape),
    )


def unpack_bool_mask(
    packed: PackedBoolMask,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Unpack a packed boolean mask into the requested device/dtype tensor."""
    mask_np = np.unpackbits(packed.packed.detach().cpu().numpy(), bitorder="little")
    total = int(np.prod(packed.shape))
    mask_np = mask_np[:total].reshape(packed.shape)
    mask = torch.from_numpy(mask_np.copy()).to(dtype=dtype)
    if device is not None:
        mask = mask.to(device)
    return mask


def packed_ternary_linear(
    x: torch.Tensor,
    positive_mask: PackedBoolMask,
    negative_mask: PackedBoolMask,
    scale: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Packed-mask ternary linear baseline. Unpacks on demand, then calls ternary_linear."""
    dev = device if device is not None else x.device
    pos = unpack_bool_mask(positive_mask, device=dev, dtype=dtype)
    neg = unpack_bool_mask(negative_mask, device=dev, dtype=dtype)
    scl = scale.to(device=dev, dtype=dtype)
    return ternary_linear(x.to(device=dev, dtype=dtype), pos, neg, scl)
