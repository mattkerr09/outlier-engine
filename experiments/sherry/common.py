from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


MODEL_REF = "Outlier-Ai/Outlier-10B"
LAYER_IDX = 8
N_EXPERTS = 8
DROP_RATE = 0.5

OUTPUT_DIR = Path(__file__).resolve().parent
SPARSIFY_SUMMARY_PATH = OUTPUT_DIR / "sparsify_summary.json"
PACKING_SUMMARY_PATH = OUTPUT_DIR / "packing_summary.json"
RESULTS_PATH = OUTPUT_DIR / "results.md"

PROJECTIONS = (
    ("gate", "gate_W", "gate_ternary", "gate_scale"),
    ("up", "up_W", "up_ternary", "up_scale"),
    ("down", "down_W", "down_ternary", "down_scale"),
)


def resolve_model_dir(model_ref: str = MODEL_REF) -> Path:
    return Path(
        snapshot_download(
            repo_id=model_ref,
            allow_patterns=["config.json", "*.safetensors"],
        )
    )


def load_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def build_tensor_index(model_dir: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if ".mlp.shared_expert." in key or ".mlp.experts." in key:
                    index[key] = str(shard)
    return index


def load_tensor(index: Dict[str, str], key: str) -> torch.Tensor:
    shard = index.get(key)
    if shard is None:
        raise KeyError(f"Tensor key not found: {key}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def shared_key(layer_idx: int, shared_name: str) -> str:
    return f"base.model.layers.{layer_idx}.mlp.shared_expert.{shared_name}"


def load_shared_fp16(index: Dict[str, str], layer_idx: int = LAYER_IDX) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for projection, shared_name, _, _ in PROJECTIONS:
        tensors[projection] = load_tensor(index, shared_key(layer_idx, shared_name))
    return tensors


def quantize_absmean(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().mean().clamp(min=1e-8)
    ternary = (weight.float() / scale).round().clamp(-1, 1).to(torch.int8)
    return ternary, scale.reshape(1).to(torch.float16)


def iter_diverse_expert_projections(
    shared_fp16: dict[str, torch.Tensor],
    expert_idx: int,
) -> Iterator[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
    seed = LAYER_IDX * 1000 + expert_idx
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    keep_prob = 1.0 - DROP_RATE
    scale_up = 1.0 / keep_prob

    for projection, _, _, _ in PROJECTIONS:
        shared_weight = shared_fp16[projection].float()
        mask = (torch.rand(shared_weight.shape, generator=generator) < keep_prob).to(torch.float32)
        upcycled = shared_weight * mask * scale_up
        ternary, q_scale = quantize_absmean(upcycled)
        yield projection, upcycled, ternary, q_scale


def zero_histogram(zero_counts: torch.Tensor) -> dict[str, int]:
    return {
        str(zero_count): int((zero_counts == zero_count).sum().item())
        for zero_count in range(5)
    }


def merge_histograms(accum: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    merged = dict(accum)
    for key, value in update.items():
        merged[key] = merged.get(key, 0) + int(value)
    return merged


def enforce_sherry_projection(
    ternary: torch.Tensor,
    original_float: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    flat_ternary = ternary.reshape(-1)
    flat_abs = original_float.abs().reshape(-1)

    pad_len = (-flat_ternary.numel()) % 4
    if pad_len:
        flat_ternary = torch.cat(
            [flat_ternary, torch.zeros(pad_len, dtype=flat_ternary.dtype, device=flat_ternary.device)]
        )
        flat_abs = torch.cat(
            [flat_abs, torch.full((pad_len,), float("inf"), dtype=flat_abs.dtype, device=flat_abs.device)]
        )

    groups = flat_ternary.view(-1, 4).clone()
    group_abs = flat_abs.view(-1, 4)

    zero_counts_before = (groups == 0).sum(dim=1)
    all_nonzero = zero_counts_before == 0
    argmin_idx = group_abs.argmin(dim=1)
    if bool(all_nonzero.any()):
        row_idx = torch.arange(groups.shape[0], device=groups.device)[all_nonzero]
        groups[row_idx, argmin_idx[all_nonzero]] = 0

    zero_counts_after = (groups == 0).sum(dim=1)
    enforced = groups.reshape(-1)[: ternary.numel()].reshape_as(ternary).to(torch.int8)

    groups_needing_enforcement = int(all_nonzero.sum().item())
    stats = {
        "numel": int(ternary.numel()),
        "groups": int(groups.shape[0]),
        "natural_zeros": int((ternary == 0).sum().item()),
        "enforced_zeros": int((enforced == 0).sum().item()),
        "groups_needing_enforcement": groups_needing_enforcement,
        "weights_forced_zero": groups_needing_enforcement,
        "zero_hist_before": zero_histogram(zero_counts_before),
        "zero_hist_after": zero_histogram(zero_counts_after),
        "exact_one_zero_groups_after": int((zero_counts_after == 1).sum().item()),
        "multi_zero_groups_after": int((zero_counts_after >= 2).sum().item()),
    }
    return enforced, stats


def tq10_nbytes(numel: int) -> int:
    return math.ceil(numel / 5)


def float16_nbytes(numel: int) -> int:
    return numel * torch.tensor([], dtype=torch.float16).element_size()


def full_expert_tq10_nbytes(expert_shapes: dict[str, int]) -> int:
    total = 0
    for projection, _, _, _ in PROJECTIONS:
        total += tq10_nbytes(expert_shapes[f"{projection}_ternary_numel"])
        total += float16_nbytes(expert_shapes[f"{projection}_scale_numel"])
    return total


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024**2)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
