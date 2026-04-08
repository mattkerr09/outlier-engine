from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from outlier_engine.paging import pack_ternary_tq10


MODEL_REF = "Outlier-Ai/Outlier-10B"
LAYER_IDX = 8
N_EXPERTS = 8
DROP_RATE = 0.5
OUTPUT_DIR = Path(__file__).resolve().parent
DIAGNOSIS_LOG_PATH = OUTPUT_DIR / "diagnosis.log"
EXPERIMENT_V2_LOG_PATH = OUTPUT_DIR / "experiment_v2_output.log"
RESULTS_PATH = OUTPUT_DIR / "results.md"
DIVERSE_EXPERTS_DIR = OUTPUT_DIR / "diverse_experts" / f"layer_{LAYER_IDX}"

PROJECTIONS = (
    ("gate", "gate_W", "gate_ternary", "gate_scale"),
    ("up", "up_W", "up_ternary", "up_scale"),
    ("down", "down_W", "down_ternary", "down_scale"),
)


class TeeLogger:
    def __init__(self, path: Path, mode: str = "w") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open(mode, encoding="utf-8")

    def emit(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


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


def load_header_entry(index: Dict[str, str], key: str) -> dict:
    shard = index.get(key)
    if shard is None:
        raise KeyError(f"Header key not found: {key}")
    with open(shard, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return {
        "shard": Path(shard).name,
        "info": header[key],
    }


def shared_key(layer_idx: int, shared_name: str) -> str:
    return f"base.model.layers.{layer_idx}.mlp.shared_expert.{shared_name}"


def expert_key(layer_idx: int, expert_idx: int, tensor_name: str) -> str:
    return f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{tensor_name}"


def quantize_absmean(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().mean().clamp(min=1e-8)
    ternary = (weight.float() / scale).round().clamp(-1, 1).to(torch.int8)
    return ternary, scale.reshape(1).to(torch.float16)


def load_shared_fp16(index: Dict[str, str], layer_idx: int = LAYER_IDX) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for projection, shared_name, _, _ in PROJECTIONS:
        tensors[projection] = load_tensor(index, shared_key(layer_idx, shared_name))
    return tensors


def quantize_shared(index: Dict[str, str], layer_idx: int = LAYER_IDX) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for projection, shared_name, _, _ in PROJECTIONS:
        ternary, scale = quantize_absmean(load_tensor(index, shared_key(layer_idx, shared_name)).float())
        tensors[f"{projection}_ternary"] = ternary
        tensors[f"{projection}_scale"] = scale
    return tensors


def load_checkpoint_expert(index: Dict[str, str], layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for projection, _, ternary_name, scale_name in PROJECTIONS:
        tensors[f"{projection}_ternary"] = load_tensor(index, expert_key(layer_idx, expert_idx, ternary_name))
        tensors[f"{projection}_scale"] = load_tensor(index, expert_key(layer_idx, expert_idx, scale_name))
    return tensors


def save_safetensors_file(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load_safetensors_file(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


def diverse_expert_path(expert_idx: int) -> Path:
    return DIVERSE_EXPERTS_DIR / f"expert_{expert_idx}.safetensors"


def shared_quantized_path() -> Path:
    return DIVERSE_EXPERTS_DIR / "shared_ternary.safetensors"


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024**2)


def tq10_nbytes(numel: int) -> int:
    return math.ceil(numel / 5)


def float16_nbytes(numel: int) -> int:
    return numel * torch.tensor([], dtype=torch.float16).element_size()


def full_expert_tq10_nbytes(expert_tensors: dict[str, torch.Tensor]) -> int:
    total = 0
    for projection, _, _, _ in PROJECTIONS:
        total += tq10_nbytes(expert_tensors[f"{projection}_ternary"].numel())
        total += float16_nbytes(expert_tensors[f"{projection}_scale"].numel())
    return total


def nonzero_stream_tq10_nbytes(values: torch.Tensor) -> int:
    if values.numel() == 0:
        return 0
    return pack_ternary_tq10(values.to(torch.int8)).numel()


def position_index_bits(numel: int) -> int:
    return max(1, math.ceil(math.log2(max(2, numel))))


def pack_bits_to_bytes(nbits: int) -> int:
    return math.ceil(nbits / 8)


def expert_cosine_similarity(
    expert_a: dict[str, torch.Tensor],
    expert_b: dict[str, torch.Tensor],
) -> float:
    dot_sum = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for projection, _, _, _ in PROJECTIONS:
        a = expert_a[f"{projection}_ternary"].reshape(-1).float()
        b = expert_b[f"{projection}_ternary"].reshape(-1).float()
        dot_sum += (a * b).sum().item()
        norm_a += a.square().sum().item()
        norm_b += b.square().sum().item()
    return dot_sum / max(math.sqrt(norm_a) * math.sqrt(norm_b), 1e-12)
