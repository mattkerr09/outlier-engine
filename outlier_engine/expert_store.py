"""
ExpertStore — single-file 4KB-aligned monolith for packed TQ1_0 experts.

File format:

  HEADER (4096 bytes, offset 0):
    magic        4 bytes   b"OEXS"
    version      uint32    1
    num_layers   uint32    e.g. 28
    experts_per_layer  uint32  e.g. 8
    num_entries  uint32    num_layers * experts_per_layer
    n_sub_files  uint32    6 (gate_t, gate_s, up_t, up_s, down_t, down_s)
    sub_sizes    6×uint64  byte size of each sub-file within an expert blob
    padding      to 4096

  INDEX TABLE (offset 4096, length = ceil(num_entries * 32 / 4096) * 4096):
    For each expert, sorted by (layer, expert_id):
      layer_id   uint64
      expert_id  uint64
      offset     uint64   absolute byte offset into the file
      size       uint64   expert blob size (before padding)

  EXPERT DATA (layer-interleaved):
    For each (layer, expert) in order:
      gate_ternary bytes || gate_scale bytes ||
      up_ternary bytes   || up_scale bytes   ||
      down_ternary bytes || down_scale bytes
      Padded to 4096-byte boundary with zeros.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import List, Optional

MAGIC = b"OEXS"
VERSION = 1
HEADER_SIZE = 4096
ALIGNMENT = 4096
N_SUB_FILES = 6
# Canonical order of sub-files within an expert blob
SUB_FILE_ORDER = [
    "gate_ternary",
    "gate_scale",
    "up_ternary",
    "up_scale",
    "down_ternary",
    "down_scale",
]


def _align(offset: int, alignment: int = ALIGNMENT) -> int:
    """Round up to next alignment boundary."""
    return (offset + alignment - 1) // alignment * alignment


def _key_for(layer: int, expert: int, sub: str) -> str:
    """Return the packed_experts index key for a sub-file."""
    return f"base.model.layers.{layer}.mlp.experts.{expert}.{sub}"


class ExpertStore:
    """Pack / load experts from a single monolith binary."""

    @staticmethod
    def pack(expert_dir: str | Path, output_path: str | Path) -> dict:
        """
        Read all individual expert files and write them into a single binary.

        Args:
            expert_dir:  directory containing *.bin files + index.json
            output_path: where to write the monolith file

        Returns:
            dict with packing stats (num_entries, total_size, etc.)
        """
        expert_dir = Path(expert_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the index to discover files and their metadata
        index = json.loads((expert_dir / "index.json").read_text(encoding="utf-8"))

        # Discover layers and experts
        import re
        entries = set()
        for k in index:
            m = re.match(
                r"base\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.", k
            )
            if m:
                entries.add((int(m.group(1)), int(m.group(2))))

        layers = sorted({l for l, _ in entries})
        experts_per_layer = sorted({e for _, e in entries})
        num_layers = len(layers)
        n_experts = len(experts_per_layer)
        num_entries = num_layers * n_experts

        # Determine sub-file sizes from the first expert (all experts are same size)
        sub_sizes = []
        for sub in SUB_FILE_ORDER:
            key = _key_for(layers[0], experts_per_layer[0], sub)
            info = index[key]
            sub_sizes.append(info["packed_bytes"])

        expert_blob_size = sum(sub_sizes)

        # Compute layout
        index_bytes = num_entries * 32  # 4 × uint64 per entry
        index_padded = _align(index_bytes)
        data_start = HEADER_SIZE + index_padded
        expert_padded_size = _align(expert_blob_size)

        with open(output_path, "wb") as f:
            # --- Write header ---
            header = bytearray(HEADER_SIZE)
            offset = 0
            # magic
            header[offset:offset + 4] = MAGIC
            offset += 4
            # version
            struct.pack_into("<I", header, offset, VERSION)
            offset += 4
            # num_layers
            struct.pack_into("<I", header, offset, num_layers)
            offset += 4
            # experts_per_layer
            struct.pack_into("<I", header, offset, n_experts)
            offset += 4
            # num_entries
            struct.pack_into("<I", header, offset, num_entries)
            offset += 4
            # n_sub_files
            struct.pack_into("<I", header, offset, N_SUB_FILES)
            offset += 4
            # sub_sizes (6 × uint64)
            for sz in sub_sizes:
                struct.pack_into("<Q", header, offset, sz)
                offset += 8

            f.write(bytes(header))

            # --- Write index table ---
            index_buf = bytearray(index_padded)
            idx_offset = 0
            data_offset = data_start
            entry_order = []

            for layer in layers:
                for expert in experts_per_layer:
                    struct.pack_into("<Q", index_buf, idx_offset, layer)
                    idx_offset += 8
                    struct.pack_into("<Q", index_buf, idx_offset, expert)
                    idx_offset += 8
                    struct.pack_into("<Q", index_buf, idx_offset, data_offset)
                    idx_offset += 8
                    struct.pack_into("<Q", index_buf, idx_offset, expert_blob_size)
                    idx_offset += 8
                    entry_order.append((layer, expert, data_offset))
                    data_offset += expert_padded_size

            f.write(bytes(index_buf))

            # --- Write expert data (layer-interleaved) ---
            for layer, expert, _ in entry_order:
                blob = bytearray()
                for sub in SUB_FILE_ORDER:
                    key = _key_for(layer, expert, sub)
                    info = index[key]
                    file_path = expert_dir / info["file"]
                    blob.extend(file_path.read_bytes())
                # Pad to alignment
                pad_needed = expert_padded_size - len(blob)
                if pad_needed > 0:
                    blob.extend(b"\x00" * pad_needed)
                f.write(bytes(blob))

        total_size = data_start + num_entries * expert_padded_size
        return {
            "num_layers": num_layers,
            "experts_per_layer": n_experts,
            "num_entries": num_entries,
            "expert_blob_size": expert_blob_size,
            "expert_padded_size": expert_padded_size,
            "total_size": total_size,
            "sub_sizes": sub_sizes,
        }

    @staticmethod
    def _read_header(f) -> dict:
        """Read and parse the header from an open file handle."""
        f.seek(0)
        raw = f.read(HEADER_SIZE)
        if raw[:4] != MAGIC:
            raise ValueError(f"Bad magic: {raw[:4]!r}, expected {MAGIC!r}")
        version = struct.unpack_from("<I", raw, 4)[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        num_layers = struct.unpack_from("<I", raw, 8)[0]
        experts_per_layer = struct.unpack_from("<I", raw, 12)[0]
        num_entries = struct.unpack_from("<I", raw, 16)[0]
        n_sub = struct.unpack_from("<I", raw, 20)[0]
        sub_sizes = []
        for i in range(n_sub):
            sub_sizes.append(struct.unpack_from("<Q", raw, 24 + i * 8)[0])
        return {
            "num_layers": num_layers,
            "experts_per_layer": experts_per_layer,
            "num_entries": num_entries,
            "sub_sizes": sub_sizes,
        }

    @staticmethod
    def _read_index(f, num_entries: int) -> dict:
        """Read index table. Returns {(layer, expert): (offset, size)}."""
        f.seek(HEADER_SIZE)
        index_bytes = num_entries * 32
        raw = f.read(index_bytes)
        table = {}
        for i in range(num_entries):
            base = i * 32
            layer = struct.unpack_from("<Q", raw, base)[0]
            expert = struct.unpack_from("<Q", raw, base + 8)[0]
            offset = struct.unpack_from("<Q", raw, base + 16)[0]
            size = struct.unpack_from("<Q", raw, base + 24)[0]
            table[(layer, expert)] = (offset, size)
        return table

    @staticmethod
    def load_expert(store_path: str | Path, layer_id: int, expert_id: int) -> bytes:
        """Load a single expert's raw bytes from the monolith."""
        with open(store_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])
            key = (layer_id, expert_id)
            if key not in idx:
                raise KeyError(f"Expert ({layer_id}, {expert_id}) not in store")
            offset, size = idx[key]
            f.seek(offset)
            return f.read(size)

    @staticmethod
    def load_layer(store_path: str | Path, layer_id: int) -> list[bytes]:
        """
        Load all experts for one layer in one sequential read.

        Returns a list of expert blobs ordered by expert_id.
        """
        with open(store_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])

            layer_entries = [
                (eid, offset, size)
                for (lid, eid), (offset, size) in idx.items()
                if lid == layer_id
            ]
            layer_entries.sort(key=lambda x: x[0])

            if not layer_entries:
                raise KeyError(f"Layer {layer_id} not in store")

            # Since experts are layer-interleaved, all layer experts are contiguous.
            # One sequential read covers them all.
            first_offset = layer_entries[0][1]
            last = layer_entries[-1]
            total_read = (last[1] + _align(last[2])) - first_offset

            f.seek(first_offset)
            bulk = f.read(total_read)

            results = []
            for eid, offset, size in layer_entries:
                local_offset = offset - first_offset
                results.append(bulk[local_offset:local_offset + size])
            return results

    @staticmethod
    def verify(
        store_path: str | Path,
        expert_dir: str | Path,
    ) -> dict:
        """
        Verify every expert in the monolith matches the individual files.

        Returns dict with verification stats.
        """
        expert_dir = Path(expert_dir)
        index = json.loads((expert_dir / "index.json").read_text(encoding="utf-8"))

        with open(store_path, "rb") as f:
            hdr = ExpertStore._read_header(f)
            idx = ExpertStore._read_index(f, hdr["num_entries"])

        checked = 0
        mismatches = 0

        for (layer, expert), (offset, size) in sorted(idx.items()):
            # Build expected blob from individual files
            expected = bytearray()
            for sub in SUB_FILE_ORDER:
                key = _key_for(layer, expert, sub)
                info = index[key]
                file_path = expert_dir / info["file"]
                expected.extend(file_path.read_bytes())

            # Load from monolith
            actual = ExpertStore.load_expert(store_path, layer, expert)

            if bytes(expected) != actual:
                mismatches += 1
            checked += 1

        return {
            "checked": checked,
            "mismatches": mismatches,
            "ok": mismatches == 0,
        }
