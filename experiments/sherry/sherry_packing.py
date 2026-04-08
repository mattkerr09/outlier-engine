from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from common import (
    LAYER_IDX,
    MODEL_REF,
    N_EXPERTS,
    PACKING_SUMMARY_PATH,
    PROJECTIONS,
    SPARSIFY_SUMMARY_PATH,
    bytes_to_mib,
    build_tensor_index,
    enforce_sherry_projection,
    float16_nbytes,
    iter_diverse_expert_projections,
    load_config,
    load_shared_fp16,
    resolve_model_dir,
    tq10_nbytes,
    write_json,
)


OUTLIER_150B_SCALE = 15.0


class BitWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.bit_length = 0

    def write_bits(self, value: int, nbits: int) -> None:
        for bit_idx in range(nbits):
            bit = (value >> bit_idx) & 1
            byte_idx = self.bit_length // 8
            offset = self.bit_length % 8
            if byte_idx == len(self.buffer):
                self.buffer.append(0)
            if bit:
                self.buffer[byte_idx] |= 1 << offset
            self.bit_length += 1

    def finish(self) -> bytes:
        return bytes(self.buffer)


class BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.bit_offset = 0

    def read_bits(self, nbits: int) -> int:
        value = 0
        for bit_idx in range(nbits):
            byte_idx = self.bit_offset // 8
            offset = self.bit_offset % 8
            bit = (self.payload[byte_idx] >> offset) & 1
            value |= bit << bit_idx
            self.bit_offset += 1
        return value


def _pad_groups(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = tensor.reshape(-1).to(torch.int8)
    pad_len = (-flat.numel()) % 4
    if pad_len:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.int8)])
    return flat.view(-1, 4), pad_len


def pack_sherry_strict(tensor: torch.Tensor) -> dict[str, object]:
    groups, pad_len = _pad_groups(tensor)
    if pad_len:
        raise ValueError("strict Sherry packing requires a group-aligned tensor")
    writer = BitWriter()
    for group in groups.tolist():
        zero_positions = [idx for idx, value in enumerate(group) if value == 0]
        if len(zero_positions) != 1:
            raise ValueError("strict Sherry packing requires exactly one zero per group")
        zero_pos = zero_positions[0]
        writer.write_bits(zero_pos, 2)
        for idx, value in enumerate(group):
            if idx == zero_pos:
                continue
            if value not in (-1, 1):
                raise ValueError("strict Sherry packing requires +/-1 in nonzero slots")
            writer.write_bits(1 if value > 0 else 0, 1)
    return {
        "data": writer.finish(),
        "bit_length": writer.bit_length,
        "numel": tensor.numel(),
        "shape": list(tensor.shape),
        "groups": groups.shape[0],
    }


def unpack_sherry_strict(payload: bytes, shape: tuple[int, ...], numel: int) -> torch.Tensor:
    reader = BitReader(payload)
    groups = math.ceil(numel / 4)
    decoded = torch.empty(groups, 4, dtype=torch.int8)
    for group_idx in range(groups):
        zero_pos = reader.read_bits(2)
        sign_values = []
        for _ in range(3):
            sign_values.append(1 if reader.read_bits(1) else -1)
        sign_idx = 0
        for slot_idx in range(4):
            if slot_idx == zero_pos:
                decoded[group_idx, slot_idx] = 0
            else:
                decoded[group_idx, slot_idx] = sign_values[sign_idx]
                sign_idx += 1
    return decoded.reshape(-1)[:numel].reshape(shape)


def pack_sherry_hybrid(tensor: torch.Tensor) -> dict[str, object]:
    groups, _ = _pad_groups(tensor)
    writer = BitWriter()
    exact_groups = 0
    fallback_groups = 0
    fallback_bits = 0

    for group in groups.tolist():
        zero_positions = [idx for idx, value in enumerate(group) if value == 0]
        if len(zero_positions) == 1:
            exact_groups += 1
            zero_pos = zero_positions[0]
            writer.write_bits(0, 1)
            writer.write_bits(zero_pos, 2)
            for idx, value in enumerate(group):
                if idx == zero_pos:
                    continue
                if value not in (-1, 1):
                    raise ValueError("strict hybrid fast path expects +/-1 in nonzero slots")
                writer.write_bits(1 if value > 0 else 0, 1)
            continue

        if len(zero_positions) == 0:
            raise ValueError("hybrid packing expects tensors already enforced to at least one zero per group")

        fallback_groups += 1
        writer.write_bits(1, 1)
        mask = 0
        sign_values: list[int] = []
        for idx, value in enumerate(group):
            if value != 0:
                mask |= 1 << idx
                sign_values.append(1 if value > 0 else 0)
        writer.write_bits(mask, 4)
        for sign in sign_values:
            writer.write_bits(sign, 1)
        fallback_bits += 1 + 4 + len(sign_values)

    return {
        "data": writer.finish(),
        "bit_length": writer.bit_length,
        "numel": tensor.numel(),
        "shape": list(tensor.shape),
        "groups": groups.shape[0],
        "exact_groups": exact_groups,
        "fallback_groups": fallback_groups,
        "fallback_bits": fallback_bits,
    }


def unpack_sherry_hybrid(payload: bytes, shape: tuple[int, ...], numel: int) -> torch.Tensor:
    reader = BitReader(payload)
    groups = math.ceil(numel / 4)
    decoded = torch.zeros(groups, 4, dtype=torch.int8)
    for group_idx in range(groups):
        mode = reader.read_bits(1)
        if mode == 0:
            zero_pos = reader.read_bits(2)
            sign_values = [1 if reader.read_bits(1) else -1 for _ in range(3)]
            sign_idx = 0
            for slot_idx in range(4):
                if slot_idx == zero_pos:
                    continue
                decoded[group_idx, slot_idx] = sign_values[sign_idx]
                sign_idx += 1
            continue

        mask = reader.read_bits(4)
        active = [slot_idx for slot_idx in range(4) if (mask >> slot_idx) & 1]
        for slot_idx in active:
            decoded[group_idx, slot_idx] = 1 if reader.read_bits(1) else -1

    return decoded.reshape(-1)[:numel].reshape(shape)


def build_strict_demo_tensor(num_groups: int = 256) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(260107892)
    groups = torch.randint(0, 2, (num_groups, 4), generator=generator, dtype=torch.int64)
    groups = groups.mul(2).sub(1).to(torch.int8)
    zero_positions = torch.randint(0, 4, (num_groups,), generator=generator, dtype=torch.int64)
    groups[torch.arange(num_groups), zero_positions] = 0
    return groups.reshape(num_groups * 4)


def average(results: list[dict], key: str) -> float:
    return sum(item[key] for item in results) / len(results)


def maybe_load_engine_expert_bytes() -> int | None:
    meta_path = Path.home() / "outlier-engine" / "packed_experts" / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(round(float(meta["packed_mb"]) * 1024**2))


def load_sparsify_summary() -> dict:
    return json.loads(SPARSIFY_SUMMARY_PATH.read_text(encoding="utf-8"))


def evaluate_expert_from_summary(expert_summary: dict, expert_shapes: dict[str, int]) -> dict:
    tq10_payload_bytes = 0
    tq10_total_bytes = 0
    sherry_ideal_bits = 0
    sherry_ideal_payload_bytes = 0
    scales_bytes = 0

    for projection, _, _, _ in PROJECTIONS:
        proj_numel = int(expert_shapes[f"{projection}_ternary_numel"])
        proj_groups = math.ceil(proj_numel / 4)
        scale_bytes = float16_nbytes(int(expert_shapes[f"{projection}_scale_numel"]))
        tq10_payload_bytes += tq10_nbytes(proj_numel)
        tq10_total_bytes += tq10_nbytes(proj_numel) + scale_bytes
        sherry_ideal_bits += proj_groups * 5
        sherry_ideal_payload_bytes += math.ceil((proj_groups * 5) / 8)
        scales_bytes += scale_bytes

    hist = {int(key): int(value) for key, value in expert_summary["zero_hist_after"].items()}
    hybrid_bits = (
        hist.get(1, 0) * 6
        + hist.get(2, 0) * 7
        + hist.get(3, 0) * 6
        + hist.get(4, 0) * 5
    )
    hybrid_payload_bytes = math.ceil(hybrid_bits / 8)
    total_numel = int(expert_summary["numel"])

    return {
        "expert_idx": expert_summary["expert_idx"],
        "numel": total_numel,
        "groups": int(expert_summary["groups"]),
        "exact_one_zero_groups_after_pct": float(expert_summary["exact_one_zero_groups_after_pct"]),
        "multi_zero_groups_after_pct": float(expert_summary["multi_zero_groups_after_pct"]),
        "tq10_payload_bytes": tq10_payload_bytes,
        "tq10_total_bytes": tq10_total_bytes,
        "tq10_bits_per_weight": (tq10_payload_bytes * 8) / max(total_numel, 1),
        "sherry_ideal_bits_per_weight": sherry_ideal_bits / max(total_numel, 1),
        "sherry_ideal_payload_bytes": sherry_ideal_payload_bytes,
        "sherry_ideal_total_bytes": sherry_ideal_payload_bytes + scales_bytes,
        "hybrid_bits_per_weight": hybrid_bits / max(total_numel, 1),
        "hybrid_payload_bytes": hybrid_payload_bytes,
        "hybrid_total_bytes": hybrid_payload_bytes + scales_bytes,
        "ideal_size_ratio_vs_tq10": (sherry_ideal_payload_bytes + scales_bytes) / max(tq10_total_bytes, 1),
        "hybrid_size_ratio_vs_tq10": (hybrid_payload_bytes + scales_bytes) / max(tq10_total_bytes, 1),
        "ideal_reduction_pct_vs_tq10": 1.0 - ((sherry_ideal_payload_bytes + scales_bytes) / max(tq10_total_bytes, 1)),
        "hybrid_reduction_pct_vs_tq10": 1.0 - ((hybrid_payload_bytes + scales_bytes) / max(tq10_total_bytes, 1)),
        "hybrid_group_histogram": hist,
    }


def build_sample_enforced_tensor(shared_fp16: dict, expert_idx: int = 0, max_groups: int = 4096) -> torch.Tensor:
    projection_name = PROJECTIONS[0][0]
    for projection, original_float, ternary, _scale in iter_diverse_expert_projections(shared_fp16, expert_idx):
        if projection != projection_name:
            continue
        enforced, _stats = enforce_sherry_projection(ternary, original_float)
        sample_numel = min(enforced.numel(), max_groups * 4)
        return enforced.reshape(-1)[:sample_numel].clone()
    raise RuntimeError("failed to build sample enforced tensor")


def main() -> None:
    strict_demo = build_strict_demo_tensor()
    strict_packed = pack_sherry_strict(strict_demo)
    strict_unpacked = unpack_sherry_strict(strict_packed["data"], tuple(strict_demo.shape), strict_demo.numel())
    strict_roundtrip_ok = bool(torch.equal(strict_demo, strict_unpacked))
    if not strict_roundtrip_ok:
        raise AssertionError("strict Sherry roundtrip failed")

    sparsify_summary = load_sparsify_summary()
    model_dir = resolve_model_dir(MODEL_REF)
    config = load_config(model_dir)
    tensor_index = build_tensor_index(model_dir)
    shared_fp16 = load_shared_fp16(tensor_index, layer_idx=LAYER_IDX)

    sample_tensor = build_sample_enforced_tensor(shared_fp16, expert_idx=0)
    sample_hybrid = pack_sherry_hybrid(sample_tensor)
    sample_unpacked = unpack_sherry_hybrid(sample_hybrid["data"], tuple(sample_tensor.shape), sample_tensor.numel())
    hybrid_roundtrip_ok = bool(torch.equal(sample_tensor, sample_unpacked))
    if not hybrid_roundtrip_ok:
        raise AssertionError("hybrid Sherry roundtrip failed on sample tensor")

    expert_shapes = sparsify_summary["expert_shapes"]
    per_expert = [
        evaluate_expert_from_summary(expert_summary, expert_shapes)
        for expert_summary in sparsify_summary["per_expert"]
    ]
    for result in per_expert:
        print(
            f"expert_{result['expert_idx']}: exact_one_zero={result['exact_one_zero_groups_after_pct'] * 100:.2f}% | "
            f"hybrid_bits={result['hybrid_bits_per_weight']:.4f} | "
            f"hybrid_ratio={result['hybrid_size_ratio_vs_tq10']:.4f} | "
            f"hybrid_reduction={result['hybrid_reduction_pct_vs_tq10'] * 100:.2f}%"
        )

    layer_tq10_total = sum(item["tq10_total_bytes"] for item in per_expert)
    layer_ideal_total = sum(item["sherry_ideal_total_bytes"] for item in per_expert)
    layer_hybrid_total = sum(item["hybrid_total_bytes"] for item in per_expert)

    current_engine_expert_bytes = maybe_load_engine_expert_bytes()
    current_engine_expert_mib = bytes_to_mib(current_engine_expert_bytes) if current_engine_expert_bytes else None
    hybrid_engine_expert_bytes = None
    ideal_engine_expert_bytes = None
    if current_engine_expert_bytes:
        hybrid_ratio = layer_hybrid_total / max(layer_tq10_total, 1)
        ideal_ratio = layer_ideal_total / max(layer_tq10_total, 1)
        hybrid_engine_expert_bytes = int(round(current_engine_expert_bytes * hybrid_ratio))
        ideal_engine_expert_bytes = int(round(current_engine_expert_bytes * ideal_ratio))

    summary = {
        "model_ref": MODEL_REF,
        "layer_idx": LAYER_IDX,
        "n_experts": N_EXPERTS,
        "strict_roundtrip_ok": strict_roundtrip_ok,
        "hybrid_roundtrip_ok": hybrid_roundtrip_ok,
        "strict_demo_bits_per_weight": strict_packed["bit_length"] / max(strict_demo.numel(), 1),
        "strict_demo_payload_bytes": len(strict_packed["data"]),
        "hybrid_sample_bits_per_weight": sample_hybrid["bit_length"] / max(sample_tensor.numel(), 1),
        "hybrid_sample_payload_bytes": len(sample_hybrid["data"]),
        "sample_expert": per_expert[0],
        "layer_8_average": {
            "exact_one_zero_groups_after_pct": average(per_expert, "exact_one_zero_groups_after_pct"),
            "multi_zero_groups_after_pct": average(per_expert, "multi_zero_groups_after_pct"),
            "tq10_bits_per_weight": average(per_expert, "tq10_bits_per_weight"),
            "sherry_ideal_bits_per_weight": average(per_expert, "sherry_ideal_bits_per_weight"),
            "hybrid_bits_per_weight": average(per_expert, "hybrid_bits_per_weight"),
            "ideal_size_ratio_vs_tq10": average(per_expert, "ideal_size_ratio_vs_tq10"),
            "hybrid_size_ratio_vs_tq10": average(per_expert, "hybrid_size_ratio_vs_tq10"),
            "ideal_reduction_pct_vs_tq10": average(per_expert, "ideal_reduction_pct_vs_tq10"),
            "hybrid_reduction_pct_vs_tq10": average(per_expert, "hybrid_reduction_pct_vs_tq10"),
        },
        "layer_8_totals": {
            "tq10_total_bytes": layer_tq10_total,
            "sherry_ideal_total_bytes": layer_ideal_total,
            "hybrid_total_bytes": layer_hybrid_total,
            "tq10_total_mib": bytes_to_mib(layer_tq10_total),
            "sherry_ideal_total_mib": bytes_to_mib(layer_ideal_total),
            "hybrid_total_mib": bytes_to_mib(layer_hybrid_total),
        },
        "all_layers_estimate": {
            "num_hidden_layers": config.get("num_hidden_layers"),
            "tq10_total_bytes": layer_tq10_total * int(config.get("num_hidden_layers", 0)),
            "sherry_ideal_total_bytes": layer_ideal_total * int(config.get("num_hidden_layers", 0)),
            "hybrid_total_bytes": layer_hybrid_total * int(config.get("num_hidden_layers", 0)),
        },
        "engine_expert_store_estimate": {
            "current_tq10_bytes": current_engine_expert_bytes,
            "current_tq10_mib": current_engine_expert_mib,
            "sherry_ideal_bytes": ideal_engine_expert_bytes,
            "sherry_ideal_mib": bytes_to_mib(ideal_engine_expert_bytes) if ideal_engine_expert_bytes else None,
            "hybrid_bytes": hybrid_engine_expert_bytes,
            "hybrid_mib": bytes_to_mib(hybrid_engine_expert_bytes) if hybrid_engine_expert_bytes else None,
            "hybrid_bytes_saved": (
                current_engine_expert_bytes - hybrid_engine_expert_bytes
                if current_engine_expert_bytes and hybrid_engine_expert_bytes
                else None
            ),
            "projected_150b_hybrid_bytes_saved": (
                int(round((current_engine_expert_bytes - hybrid_engine_expert_bytes) * OUTLIER_150B_SCALE))
                if current_engine_expert_bytes and hybrid_engine_expert_bytes
                else None
            ),
        },
        "per_expert": per_expert,
    }

    write_json(PACKING_SUMMARY_PATH, summary)
    print(f"wrote {PACKING_SUMMARY_PATH}")
    print(json.dumps(summary["layer_8_average"], indent=2))


if __name__ == "__main__":
    main()
