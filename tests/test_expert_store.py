"""Unit tests for ExpertStore monolith packing."""

import json
import os
import struct
import tempfile
from pathlib import Path

import pytest

from outlier_engine.expert_store import (
    ALIGNMENT,
    HEADER_SIZE,
    MAGIC,
    SUB_FILE_ORDER,
    VERSION,
    ExpertStore,
    _align,
    _key_for,
)

EXPERT_DIR = Path.home() / "outlier-engine" / "packed_experts"

# Skip all tests if packed experts aren't available
pytestmark = pytest.mark.skipif(
    not (EXPERT_DIR / "index.json").exists(),
    reason="packed_experts directory not found",
)


@pytest.fixture(scope="module")
def monolith_path(tmp_path_factory):
    """Pack experts into a temporary monolith once for the entire module."""
    tmp = tmp_path_factory.mktemp("monolith")
    path = tmp / "test_experts.bin"
    ExpertStore.pack(str(EXPERT_DIR), str(path))
    return path


@pytest.fixture(scope="module")
def index():
    return json.loads((EXPERT_DIR / "index.json").read_text(encoding="utf-8"))


class TestPackUnpackRoundtrip:
    """test_pack_unpack_roundtrip: pack experts, load each, verify identical."""

    def test_all_experts_roundtrip(self, monolith_path, index):
        """Every expert loaded from monolith matches the original files byte-for-byte."""
        result = ExpertStore.verify(str(monolith_path), str(EXPERT_DIR))
        assert result["ok"], f"{result['mismatches']} mismatches out of {result['checked']}"
        assert result["checked"] == 224  # 28 layers × 8 experts

    def test_single_expert_roundtrip(self, monolith_path, index):
        """Spot-check: load expert (14, 3) and verify against original files."""
        layer, expert = 14, 3
        loaded = ExpertStore.load_expert(str(monolith_path), layer, expert)

        expected = bytearray()
        for sub in SUB_FILE_ORDER:
            key = _key_for(layer, expert, sub)
            info = index[key]
            path = EXPERT_DIR / info["file"]
            expected.extend(path.read_bytes())

        assert bytes(expected) == loaded

    def test_load_layer_roundtrip(self, monolith_path, index):
        """Load all experts for layer 8 and verify each."""
        layer = 8
        blobs = ExpertStore.load_layer(str(monolith_path), layer)
        assert len(blobs) == 8

        for expert_id, blob in enumerate(blobs):
            expected = bytearray()
            for sub in SUB_FILE_ORDER:
                key = _key_for(layer, expert_id, sub)
                info = index[key]
                path = EXPERT_DIR / info["file"]
                expected.extend(path.read_bytes())
            assert bytes(expected) == blob, f"Mismatch for expert {expert_id}"


class TestAlignment:
    """test_4kb_alignment: verify every expert offset is 4096-byte aligned."""

    def test_all_offsets_aligned(self, monolith_path):
        """Every expert data offset in the index is 4096-byte aligned."""
        with open(monolith_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])

        for (layer, expert), (offset, size) in idx.items():
            assert offset % ALIGNMENT == 0, (
                f"Expert ({layer}, {expert}) offset {offset} not {ALIGNMENT}-aligned"
            )

    def test_header_size(self, monolith_path):
        """Header is exactly 4096 bytes."""
        assert os.path.getsize(monolith_path) >= HEADER_SIZE
        with open(monolith_path, "rb") as f:
            header = f.read(HEADER_SIZE)
        assert len(header) == HEADER_SIZE
        assert header[:4] == MAGIC
        version = struct.unpack_from("<I", header, 4)[0]
        assert version == VERSION

    def test_index_table_aligned(self, monolith_path):
        """Index table starts at offset 4096 (after header)."""
        with open(monolith_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            # First entry should be right after the header
            f.seek(HEADER_SIZE)
            raw = f.read(32)
            layer = struct.unpack_from("<Q", raw, 0)[0]
            expert = struct.unpack_from("<Q", raw, 8)[0]
            assert layer == 0 and expert == 0


class TestLayerInterleaving:
    """test_layer_interleaving: verify experts are ordered by layer in the file."""

    def test_offsets_increase_by_layer(self, monolith_path):
        """Expert offsets increase monotonically: all layer-0 before layer-1, etc."""
        with open(monolith_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])

        # Sort by offset and verify layer ordering
        sorted_entries = sorted(idx.items(), key=lambda x: x[1][0])
        prev_layer = -1
        prev_offset = -1
        for (layer, expert), (offset, size) in sorted_entries:
            assert offset > prev_offset, "Offsets should be strictly increasing"
            assert layer >= prev_layer, (
                f"Layer {layer} appears after layer {prev_layer} — not interleaved"
            )
            if layer > prev_layer:
                prev_layer = layer
            prev_offset = offset

    def test_layer_experts_contiguous(self, monolith_path):
        """All experts for a given layer are contiguous in the file."""
        with open(monolith_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])

        for layer in range(28):
            layer_entries = sorted(
                [(eid, off, sz) for (lid, eid), (off, sz) in idx.items() if lid == layer],
                key=lambda x: x[1],
            )
            # Check that each expert starts right after the previous one's padded end
            for i in range(1, len(layer_entries)):
                _, prev_off, prev_sz = layer_entries[i - 1]
                _, curr_off, _ = layer_entries[i]
                expected = prev_off + _align(prev_sz)
                assert curr_off == expected, (
                    f"Layer {layer}: expert {layer_entries[i][0]} at {curr_off}, "
                    f"expected {expected} (gap or overlap)"
                )


class TestEdgeCases:
    """Additional edge case tests."""

    def test_invalid_expert_raises(self, monolith_path):
        """Loading a non-existent expert raises KeyError."""
        with pytest.raises(KeyError):
            ExpertStore.load_expert(str(monolith_path), 99, 0)

    def test_invalid_layer_raises(self, monolith_path):
        """Loading a non-existent layer raises KeyError."""
        with pytest.raises(KeyError):
            ExpertStore.load_layer(str(monolith_path), 99)

    def test_magic_bytes(self, monolith_path):
        """File starts with correct magic bytes."""
        with open(monolith_path, "rb") as f:
            assert f.read(4) == MAGIC
