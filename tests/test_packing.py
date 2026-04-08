from __future__ import annotations

import torch

from outlier_engine.paging import pack_ternary_tq10, unpack_ternary_tq10


def test_pack_unpack_roundtrip():
    tensor = torch.randint(-1, 2, (100, 100), dtype=torch.int8)

    packed = pack_ternary_tq10(tensor)
    unpacked = unpack_ternary_tq10(packed, tuple(tensor.shape))

    assert torch.equal(unpacked, tensor)


def test_pack_compression_ratio():
    tensor = torch.randint(-1, 2, (100, 100), dtype=torch.int8)

    packed = pack_ternary_tq10(tensor)
    compression = tensor.numel() / packed.numel()

    assert compression == 5.0


def test_unpack_dtype():
    tensor = torch.randint(-1, 2, (64, 64), dtype=torch.int8)

    packed = pack_ternary_tq10(tensor)
    unpacked = unpack_ternary_tq10(packed, tuple(tensor.shape))

    assert unpacked.dtype == torch.int8
