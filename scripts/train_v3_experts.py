#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from outlier_engine.loader import load_model
from outlier_engine.paging import _normalize_config


DEFAULT_MODEL_REF = "Outlier-Ai/Outlier-10B"
DEFAULT_CORPUS_DIR = Path("data/distillation")
DEFAULT_OUTPUT_DIR = Path("checkpoints/outlier-10b-v3")
TRAINING_LOG_PATH = Path("data/distillation/training_log.jsonl")

TEACHER_TEMP = 4.0
STUDENT_TEMP = 2.0
KL_CLIP = 10.0
L1_COEFF_START = 1e-4
L1_COEFF_END = 1e-3
EXPERT_STEPS = 200
ROUTER_STEPS = 100
MAX_LENGTH = 512


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def abs_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (repo_root() / p).resolve()


def log(message: str) -> None:
    print(message, flush=True)


def mps_empty_cache() -> None:
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def rss_gb() -> float:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value > 1024**3:
        return value / 1024**3
    if value > 1024**2:
        return value / 1024**2
    return value / 1024**2


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass(frozen=True)
class ExpertSpec:
    layer_idx: int
    expert_idx: int

    @property
    def global_idx(self) -> int:
        return self.layer_idx * 8 + self.expert_idx


class DistillationCorpus:
    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir
        manifest_path = corpus_dir / "corpus_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards = sorted(corpus_dir.glob("corpus_shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No corpus shards found in {corpus_dir}")
        self.examples = int(self.manifest["examples"])
        self.shard_size = int(self.manifest["shard_size"])
        self.max_length = int(self.manifest["max_length"])
        self.top_k = int(self.manifest["top_k"])
        self._cached_shard_idx: Optional[int] = None
        self._cached_shard: Optional[list[dict]] = None

    def __len__(self) -> int:
        return self.examples

    def _load_shard(self, shard_idx: int) -> list[dict]:
        if self._cached_shard_idx == shard_idx and self._cached_shard is not None:
            return self._cached_shard
        shard = torch.load(self.shards[shard_idx], map_location="cpu")
        self._cached_shard_idx = shard_idx
        self._cached_shard = shard
        return shard

    def get(self, index: int) -> dict:
        if index < 0 or index >= self.examples:
            raise IndexError(index)
        shard_idx = index // self.shard_size
        inner_idx = index % self.shard_size
        shard = self._load_shard(shard_idx)
        return shard[inner_idx]

    def iter_for_expert(self, spec: ExpertSpec, steps: int) -> Iterator[dict]:
        total = len(self)
        stride = 97
        start = (spec.global_idx * 131 + spec.layer_idx * 17) % total
        for step in range(steps):
            yield self.get((start + step * stride) % total)

    def first_n(self, n: int) -> List[dict]:
        return [self.get(i) for i in range(min(n, len(self)))]


def checkpoint_snapshot(model_ref: str) -> Path:
    return Path(
        snapshot_download(
            repo_id=model_ref,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.py",
                "*.txt",
                "*.tiktoken",
                "tokenizer*",
                "special_tokens_map.json",
                "generation_config.json",
                "chat_template.jinja",
                "README*",
            ],
        )
    ).resolve()


def load_config(model_dir: Path) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return _normalize_config(cfg)


def classify_tensor_key(key: str) -> str:
    if ".mlp.experts." in key:
        return "expert"
    if ".mlp.shared_expert." in key:
        return "shared"
    if ".mlp.router.weight" in key:
        return "router"
    return "other"


def scan_checkpoint_bytes(model_dir: Path) -> Dict[str, int]:
    counts = defaultdict(int)
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                counts[classify_tensor_key(key)] += tensor_bytes(tensor)
    return dict(counts)


def quantized_shared_bytes(cfg: dict) -> int:
    hidden = int(cfg["hidden_dim"])
    intermediate = int(cfg["intermediate_dim"])
    n_layers = int(cfg["n_layers"])
    shared_weight_bytes = n_layers * (2 * intermediate * hidden + hidden * intermediate)
    shared_scale_bytes = n_layers * (2 * intermediate + hidden) * 2
    return shared_weight_bytes + shared_scale_bytes


def expert_train_bytes(cfg: dict) -> Dict[str, int]:
    hidden = int(cfg["hidden_dim"])
    intermediate = int(cfg["intermediate_dim"])
    params = 3 * hidden * intermediate
    expert_bytes = params * 4
    return {
        "expert_fp32": expert_bytes,
        "optimizer": expert_bytes * 2,
        "gradients": expert_bytes,
    }


def distillation_batch_bytes(max_length: int, top_k: int) -> int:
    return (
        max_length * 4
        + max_length * 1
        + max_length * top_k * 2
        + max_length * top_k * 4
    )


def format_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GB"


def format_mb(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.0f} MB"


def print_memory_budget(model_dir: Path, cfg: dict, max_length: int, top_k: int) -> Dict[str, int]:
    raw = scan_checkpoint_bytes(model_dir)
    frozen_backbone = raw.get("other", 0) + raw.get("router", 0) + quantized_shared_bytes(cfg)
    expert_bytes = expert_train_bytes(cfg)
    batch_bytes = distillation_batch_bytes(max_length, top_k)
    total_est = frozen_backbone + expert_bytes["expert_fp32"] + expert_bytes["optimizer"] + expert_bytes["gradients"] + batch_bytes
    margin = 50 * 1024**3 - total_est

    log("=== MEMORY BUDGET ===")
    log(f"Frozen backbone memory: {format_gb(frozen_backbone)}")
    log(f"One expert float32 weights: {format_mb(expert_bytes['expert_fp32'])}")
    log(f"One expert optimizer states: {format_mb(expert_bytes['optimizer'])}")
    log(f"One expert gradient buffers: {format_mb(expert_bytes['gradients'])}")
    log(f"Distillation batch: {format_mb(batch_bytes)}")
    log(f"Total estimated: {format_gb(total_est)}")
    log("Available on Mac Studio: 64.00 GB")
    log(f"Margin vs 50 GB safety cap: {format_gb(max(margin, 0)) if margin >= 0 else '-' + format_gb(abs(margin))}")
    return {
        "frozen_backbone": frozen_backbone,
        "expert_fp32": expert_bytes["expert_fp32"],
        "optimizer": expert_bytes["optimizer"],
        "gradients": expert_bytes["gradients"],
        "batch": batch_bytes,
        "total_est": total_est,
        "margin": margin,
    }


def load_frozen_v1_model(model_ref: str, device: str) -> tuple[object, Path, dict]:
    loaded = load_model(model_ref, device=device, paged=True)
    model_dir = loaded.model_dir
    cfg = loaded.config
    model = loaded.model
    model.eval()
    try:
        model = model.to(dtype=torch.bfloat16)
    except Exception:
        log("WARNING: failed to cast frozen backbone to bfloat16; keeping runtime dtype as loaded.")
    for param in model.parameters():
        param.requires_grad = False
    return model, model_dir, cfg


def dequant_int8_matmul_bf16(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    chunk: int = 2048,
) -> torch.Tensor:
    out_features, in_features = weight_int8.shape
    compute_dtype = torch.float32 if x.device.type == "cpu" else torch.bfloat16
    x_comp = x.to(compute_dtype)
    s = scale.to(device=x.device, dtype=compute_dtype)
    result = torch.zeros(x.shape[0], out_features, device=x.device, dtype=compute_dtype)
    for start in range(0, in_features, chunk):
        end = min(start + chunk, in_features)
        w_chunk = weight_int8[:, start:end].to(device=x.device, dtype=compute_dtype) * s
        result += x_comp[:, start:end] @ w_chunk.T
        del w_chunk
    return result


def run_shared_swigu(shared: nn.Module, x: torch.Tensor) -> torch.Tensor:
    gate = dequant_int8_matmul_bf16(x, shared.gate_w, shared.gate_s)
    up = dequant_int8_matmul_bf16(x, shared.up_w, shared.up_s)
    hidden = F.silu(gate) * up
    return dequant_int8_matmul_bf16(hidden, shared.down_w, shared.down_s)


class SharedOnlyMLP(nn.Module):
    def __init__(self, shared: nn.Module) -> None:
        super().__init__()
        self.shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        out = run_shared_swigu(self.shared, x_flat)
        return out.to(x.dtype).view(batch, seq_len, hidden)


class Float32Expert(nn.Module):
    def __init__(self, gate_weight: torch.Tensor, up_weight: torch.Tensor, down_weight: torch.Tensor) -> None:
        super().__init__()
        self.gate_weight = nn.Parameter(gate_weight.float().contiguous())
        self.up_weight = nn.Parameter(up_weight.float().contiguous())
        self.down_weight = nn.Parameter(down_weight.float().contiguous())

    @classmethod
    def from_shared(cls, shared: nn.Module) -> "Float32Expert":
        gate = (shared.gate_w.float() * shared.gate_s.float()).cpu()
        up = (shared.up_w.float() * shared.up_s.float()).cpu()
        down = (shared.down_w.float() * shared.down_s.float()).cpu()
        return cls(gate, up, down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        gate = F.silu(F.linear(x32, self.gate_weight))
        up = F.linear(x32, self.up_weight)
        return F.linear(gate * up, self.down_weight)

    def quantize_absmean(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name in ("gate_weight", "up_weight", "down_weight"):
            weight = getattr(self, name).detach().float().cpu()
            alpha = weight.abs().mean().clamp(min=1e-8)
            threshold = 0.5 * alpha
            ternary = torch.where(
                weight > threshold,
                torch.ones_like(weight),
                torch.where(weight < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
            ).to(torch.int8)
            scale = alpha.reshape(1).to(torch.float16)
            prefix = name.replace("_weight", "")
            out[f"{prefix}_ternary"] = ternary
            out[f"{prefix}_scale"] = scale
        return out


class SingleActiveExpertMLP(nn.Module):
    def __init__(self, shared: nn.Module, expert: Float32Expert) -> None:
        super().__init__()
        self.shared = shared
        self.expert = expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        shared_out = run_shared_swigu(self.shared, x_flat).float()
        expert_out = self.expert(x_flat)
        return (shared_out + expert_out).to(x.dtype).view(batch, seq_len, hidden)


class LocalExpertPageManager:
    def __init__(self, experts_dir: Path, device: torch.device, hot_capacity: int = 64) -> None:
        self.experts_dir = experts_dir
        self.device = device
        self.hot_capacity = hot_capacity
        self.hot_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self.hot_lru: List[Tuple[int, int]] = []
        self.hot_hits = 0
        self.warm_hits = 0
        self.cold_misses = 0
        self.disk_load_s = 0.0
        self.lookups = 0

    def _file_for(self, layer_idx: int, expert_idx: int) -> Path:
        return self.experts_dir / f"layer_{layer_idx:02d}_expert_{expert_idx:02d}.safetensors"

    def _remember(self, key: Tuple[int, int], tensors: Dict[str, torch.Tensor]) -> None:
        if key in self.hot_cache:
            self.hot_lru.remove(key)
        elif len(self.hot_lru) >= self.hot_capacity:
            evicted = self.hot_lru.pop(0)
            self.hot_cache.pop(evicted, None)
        self.hot_cache[key] = tensors
        self.hot_lru.append(key)

    def get_expert(self, layer_idx: int, expert_idx: int) -> Dict[str, torch.Tensor]:
        key = (layer_idx, expert_idx)
        self.lookups += 1
        if key in self.hot_cache:
            self.hot_hits += 1
            self.hot_lru.remove(key)
            self.hot_lru.append(key)
            return self.hot_cache[key]

        path = self._file_for(layer_idx, expert_idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing trained expert shard: {path}")
        start = time.perf_counter()
        tensors = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key_name in handle.keys():
                tensors[key_name] = handle.get_tensor(key_name)
        self.disk_load_s += time.perf_counter() - start
        self.cold_misses += 1
        dev_dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32
        ready = {
            "gate_w": tensors["gate_ternary"].to(self.device, dtype=dev_dtype) * tensors["gate_scale"].to(self.device, dtype=dev_dtype),
            "up_w": tensors["up_ternary"].to(self.device, dtype=dev_dtype) * tensors["up_scale"].to(self.device, dtype=dev_dtype),
            "down_w": tensors["down_ternary"].to(self.device, dtype=dev_dtype) * tensors["down_scale"].to(self.device, dtype=dev_dtype),
        }
        self._remember(key, ready)
        return ready

    def stats(self) -> Dict[str, float]:
        total = self.lookups if self.lookups else 1
        return {
            "hot_hits": self.hot_hits,
            "warm_hits": self.warm_hits,
            "cold_misses": self.cold_misses,
            "hit_rate": (self.hot_hits + self.warm_hits) / total,
            "disk_load_s": self.disk_load_s,
            "hot_cache_entries": len(self.hot_cache),
        }


class RoutedTrainedExpertMLP(nn.Module):
    def __init__(self, shared: nn.Module, router_weight: torch.Tensor, layer_idx: int, top_k: int, expert_manager: LocalExpertPageManager) -> None:
        super().__init__()
        self.shared = shared
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.expert_manager = expert_manager
        self.router_weight = nn.Parameter(router_weight.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        logits = F.linear(x_flat.float(), self.router_weight.float())
        probs = F.softmax(logits, dim=-1)
        top_w, top_idx = torch.topk(probs, k=min(self.top_k, probs.shape[-1]), dim=-1)
        top_w = top_w / top_w.sum(-1, keepdim=True)

        shared_out = run_shared_swigu(self.shared, x_flat).float()
        expert_out = torch.zeros_like(shared_out)
        used_ids = [int(v) for v in torch.unique(top_idx).tolist()]
        for expert_idx in used_ids:
            assignment = top_idx == expert_idx
            token_mask = assignment.any(dim=-1)
            if not token_mask.any():
                continue
            weights = (top_w * assignment.to(top_w.dtype)).sum(dim=-1, keepdim=True)
            expert = self.expert_manager.get_expert(self.layer_idx, expert_idx)
            x_tok = x_flat[token_mask].to(expert["gate_w"].dtype)
            gate = F.silu(F.linear(x_tok, expert["gate_w"]))
            up = F.linear(x_tok, expert["up_w"])
            out = F.linear(gate * up, expert["down_w"]).float()
            expert_out[token_mask] += weights[token_mask] * out
        return (shared_out + expert_out).to(x.dtype).view(batch, seq_len, hidden)


def build_shared_only_backbone(model_ref: str, device: str) -> tuple[object, Path, dict]:
    model, model_dir, cfg = load_frozen_v1_model(model_ref, device)
    for layer in model.model.layers:
        layer.mlp = SharedOnlyMLP(layer.mlp.shared)
    model.eval()
    return model, model_dir, cfg


def make_target_model(
    base_model: object,
    layer_idx: int,
    *,
    device: torch.device,
) -> tuple[object, Float32Expert]:
    layer = base_model.model.layers[layer_idx]
    expert = Float32Expert.from_shared(layer.mlp.shared).to(device)
    layer.mlp = SingleActiveExpertMLP(layer.mlp.shared, expert).to(device)
    return base_model, expert


def restore_shared_only(base_model: object, layer_idx: int) -> None:
    layer = base_model.model.layers[layer_idx]
    if isinstance(layer.mlp, SingleActiveExpertMLP):
        layer.mlp = SharedOnlyMLP(layer.mlp.shared).to(next(base_model.parameters()).device)


def example_to_device(example: dict, device: torch.device) -> dict:
    return {
        "input_ids": example["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0),
        "attention_mask": example["attention_mask"].to(device=device, dtype=torch.float32).unsqueeze(0),
        "top_k_logits": example["top_k_logits"].to(device=device, dtype=torch.float32).unsqueeze(0),
        "top_k_indices": example["top_k_indices"].to(device=device, dtype=torch.long).unsqueeze(0),
        "domain": example.get("domain", "unknown"),
        "text_preview": example.get("text_preview", ""),
    }


def progressive_l1(step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return L1_COEFF_END
    ratio = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return L1_COEFF_START + (L1_COEFF_END - L1_COEFF_START) * ratio


def cakld_loss(student_logits: torch.Tensor, teacher_topk_logits: torch.Tensor, teacher_topk_indices: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
    gathered = torch.gather(student_logits.float(), dim=-1, index=teacher_topk_indices)
    t_probs = F.softmax(teacher_topk_logits / TEACHER_TEMP, dim=-1)
    s_log = F.log_softmax(gathered / STUDENT_TEMP, dim=-1)
    kl = F.kl_div(s_log, t_probs, reduction="none").sum(dim=-1)
    conf = t_probs.max(dim=-1).values
    masked = torch.clamp(kl, max=KL_CLIP) * conf * attention_mask
    loss = masked.sum() / attention_mask.sum().clamp_min(1.0)
    loss = loss * (STUDENT_TEMP ** 2)
    stats = {
        "confidence": conf.mean().item(),
        "teacher_entropy": -(t_probs.clamp_min(1e-8).log() * t_probs).sum(dim=-1).mean().item(),
    }
    return loss, stats


def l1_delta_loss(expert: Float32Expert, shared: nn.Module, coeff: float) -> tuple[torch.Tensor, dict]:
    deltas = [
        expert.gate_weight - (shared.gate_w.float().to(expert.gate_weight.device) * shared.gate_s.float().to(expert.gate_weight.device)),
        expert.up_weight - (shared.up_w.float().to(expert.up_weight.device) * shared.up_s.float().to(expert.up_weight.device)),
        expert.down_weight - (shared.down_w.float().to(expert.down_weight.device) * shared.down_s.float().to(expert.down_weight.device)),
    ]
    l1 = sum(delta.abs().mean() for delta in deltas) / len(deltas)
    sparsity = sum((delta.abs() < 0.01).float().mean() for delta in deltas) / len(deltas)
    flat_expert = torch.cat([expert.gate_weight.flatten(), expert.up_weight.flatten(), expert.down_weight.flatten()])
    flat_shared = torch.cat([
        (shared.gate_w.float().to(expert.gate_weight.device) * shared.gate_s.float().to(expert.gate_weight.device)).flatten(),
        (shared.up_w.float().to(expert.up_weight.device) * shared.up_s.float().to(expert.up_weight.device)).flatten(),
        (shared.down_w.float().to(expert.down_weight.device) * shared.down_s.float().to(expert.down_weight.device)).flatten(),
    ])
    cosine = F.cosine_similarity(flat_expert.unsqueeze(0), flat_shared.unsqueeze(0)).item()
    return coeff * l1, {"delta_sparsity": sparsity.item(), "cosine_to_shared": cosine}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def save_quantized_expert(output_dir: Path, spec: ExpertSpec, expert: Float32Expert) -> Path:
    experts_dir = output_dir / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)
    tensors = expert.quantize_absmean()
    path = experts_dir / f"layer_{spec.layer_idx:02d}_expert_{spec.expert_idx:02d}.safetensors"
    save_file(tensors, str(path))
    return path


def train_single_expert(
    model: object,
    corpus: DistillationCorpus,
    spec: ExpertSpec,
    output_dir: Path,
    steps_per_expert: int,
    device: torch.device,
) -> dict:
    model, expert = make_target_model(model, spec.layer_idx, device=device)
    target_mlp = model.model.layers[spec.layer_idx].mlp
    assert isinstance(target_mlp, SingleActiveExpertMLP)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-4, weight_decay=0.01)
    start = time.perf_counter()
    last_loss = 0.0
    last_stats: dict = {}

    for step_idx, raw_example in enumerate(corpus.iter_for_expert(spec, steps_per_expert), start=1):
        example = example_to_device(raw_example, device)
        optimizer.zero_grad(set_to_none=True)
        sync_device(device)
        outputs = model(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"].to(dtype=torch.long),
            use_cache=False,
        )
        student_logits = outputs.logits
        loss_main, main_stats = cakld_loss(
            student_logits,
            example["top_k_logits"],
            example["top_k_indices"],
            example["attention_mask"],
        )
        l1_coeff = progressive_l1(step_idx - 1, steps_per_expert)
        loss_reg, reg_stats = l1_delta_loss(expert, target_mlp.shared, coeff=l1_coeff)
        loss = loss_main + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = float(loss.item())
        last_stats = {**main_stats, **reg_stats, "l1_coeff": l1_coeff}
        if step_idx % 10 == 0 or step_idx == 1 or step_idx == steps_per_expert:
            peak_memory = rss_gb()
            log(
                f"[expert {spec.global_idx:03d} | layer {spec.layer_idx:02d} expert {spec.expert_idx}] "
                f"step {step_idx:03d}/{steps_per_expert} "
                f"loss={last_loss:.4f} lr=1.0e-4 sparsity={last_stats['delta_sparsity']:.3f} peak_memory={peak_memory:.2f}GB"
            )
        append_jsonl(
            TRAINING_LOG_PATH,
            {
                "phase": "expert_train",
                "layer": spec.layer_idx,
                "expert": spec.expert_idx,
                "global_expert_idx": spec.global_idx,
                "step": step_idx,
                "loss": last_loss,
                **last_stats,
            },
        )

    expert_path = save_quantized_expert(output_dir, spec, expert)
    elapsed = time.perf_counter() - start
    summary = {
        "layer": spec.layer_idx,
        "expert": spec.expert_idx,
        "global_expert_idx": spec.global_idx,
        "final_loss": last_loss,
        "delta_sparsity": last_stats.get("delta_sparsity", 0.0),
        "cosine_to_shared": last_stats.get("cosine_to_shared", 0.0),
        "time_s": elapsed,
        "expert_path": str(expert_path),
    }
    log(
        f"EXPERT COMPLETE layer={spec.layer_idx} expert={spec.expert_idx} "
        f"final_loss={summary['final_loss']:.4f} "
        f"delta_sparsity={summary['delta_sparsity']:.3f} "
        f"cosine_to_shared={summary['cosine_to_shared']:.4f} "
        f"time={elapsed/60.0:.1f}m"
    )
    append_jsonl(TRAINING_LOG_PATH, {"phase": "expert_complete", **summary})
    del optimizer
    del expert
    gc.collect()
    mps_empty_cache()
    restore_shared_only(model, spec.layer_idx)
    return summary


def build_router_tune_model(
    model_ref: str,
    experts_dir: Path,
    device: str,
) -> tuple[object, LocalExpertPageManager, dict]:
    model, _, cfg = load_frozen_v1_model(model_ref, device)
    manager = LocalExpertPageManager(experts_dir=experts_dir, device=torch.device(device), hot_capacity=64)
    for layer_idx, layer in enumerate(model.model.layers):
        router_weight = layer.mlp.router_weight.detach()
        layer.mlp = RoutedTrainedExpertMLP(layer.mlp.shared, router_weight, layer_idx, cfg["top_k"], manager).to(torch.device(device))
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.model.layers:
        layer.mlp.router_weight.requires_grad = True
    model.train()
    return model, manager, cfg


def train_router(
    model_ref: str,
    corpus: DistillationCorpus,
    output_dir: Path,
    device: str,
    steps: int,
) -> Dict[str, float]:
    model, manager, _ = build_router_tune_model(model_ref, output_dir / "experts", device)
    router_params = [layer.mlp.router_weight for layer in model.model.layers]
    optimizer = torch.optim.AdamW(router_params, lr=1e-3, weight_decay=0.0)
    examples = corpus.first_n(1000)
    start = time.perf_counter()
    last_loss = 0.0

    for step_idx in range(1, steps + 1):
        example = example_to_device(examples[(step_idx - 1) % len(examples)], torch.device(device))
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=example["input_ids"],
            attention_mask=example["attention_mask"].to(dtype=torch.long),
            use_cache=False,
        )
        loss, stats = cakld_loss(
            outputs.logits,
            example["top_k_logits"],
            example["top_k_indices"],
            example["attention_mask"],
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step_idx % 10 == 0 or step_idx == 1 or step_idx == steps:
            log(
                f"[router] step {step_idx:03d}/{steps} loss={last_loss:.4f} "
                f"peak_memory={rss_gb():.2f}GB hit_rate={manager.stats()['hit_rate']:.3f}"
            )
        append_jsonl(
            TRAINING_LOG_PATH,
            {
                "phase": "router_tune",
                "step": step_idx,
                "loss": last_loss,
                **stats,
                **manager.stats(),
            },
        )
    elapsed = time.perf_counter() - start
    router_dir = output_dir / "router"
    router_dir.mkdir(parents=True, exist_ok=True)
    router_state = {
        f"layer_{idx:02d}_router_weight": layer.mlp.router_weight.detach().cpu()
        for idx, layer in enumerate(model.model.layers)
    }
    save_file(router_state, str(router_dir / "router_weights.safetensors"))
    summary = {
        "router_loss": last_loss,
        "router_time_s": elapsed,
        **manager.stats(),
    }
    append_jsonl(TRAINING_LOG_PATH, {"phase": "router_complete", **summary})
    return summary


def load_router_state(router_dir: Path) -> Dict[int, torch.Tensor]:
    router_path = router_dir / "router_weights.safetensors"
    if not router_path.exists():
        return {}
    state = {}
    with safe_open(str(router_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = re.match(r"layer_(\d+)_router_weight", key)
            if match:
                state[int(match.group(1))] = handle.get_tensor(key)
    return state


def write_v3_checkpoint(base_model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    router_state = load_router_state(output_dir / "router")
    replacement_paths = {
        (int(path.stem.split("_")[1]), int(path.stem.split("_")[3])): path
        for path in (output_dir / "experts").glob("layer_*_expert_*.safetensors")
    }
    replacement_tensors: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for key, path in replacement_paths.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            replacement_tensors[key] = {name: handle.get_tensor(name) for name in handle.keys()}

    for item in base_model_dir.iterdir():
        if item.suffix == ".safetensors":
            continue
        if item.is_file():
            shutil.copy2(item, output_dir / item.name)

    for shard in sorted(base_model_dir.glob("*.safetensors")):
        state: Dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                expert_match = re.match(r"^base\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_(ternary|scale)$", key)
                if expert_match:
                    layer_idx = int(expert_match.group(1))
                    expert_idx = int(expert_match.group(2))
                    proj = expert_match.group(3)
                    suffix = expert_match.group(4)
                    repl = replacement_tensors.get((layer_idx, expert_idx))
                    if repl is not None:
                        repl_key = f"{proj}_{suffix}"
                        state[key] = repl[repl_key]
                        continue
                router_match = re.match(r"^base\.model\.layers\.(\d+)\.mlp\.router\.weight$", key)
                if router_match:
                    layer_idx = int(router_match.group(1))
                    if layer_idx in router_state:
                        state[key] = router_state[layer_idx].to(dtype=handle.get_tensor(key).dtype)
                        continue
                state[key] = handle.get_tensor(key)
        save_file(state, str(output_dir / shard.name))

    base_cfg = json.loads((base_model_dir / "config.json").read_text(encoding="utf-8"))
    base_cfg["model_type"] = "outlier_moe"
    base_cfg["v3_pipeline"] = "sequential_cakld"
    base_cfg["v3_checkpoint"] = "outlier-10b-v3"
    (output_dir / "config.json").write_text(json.dumps(base_cfg, indent=2) + "\n", encoding="utf-8")


def build_inference_model(checkpoint_dir: Path, device: str) -> object:
    loaded = load_model(str(checkpoint_dir), device=device, paged=True)
    model = loaded.model
    try:
        model = model.to(dtype=torch.bfloat16)
    except Exception:
        pass
    model.eval()
    return model


@torch.no_grad()
def run_quality_check(checkpoint_dir: Path, device: str) -> List[Tuple[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = build_inference_model(checkpoint_dir, device)
    prompts = [
        "The capital of France is",
        "Explain quantum computing:",
        "What is 15 + 27?",
        "Write Python code to sort a list:",
        "The meaning of life is",
    ]
    outputs: List[Tuple[str, str]] = []
    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        generated = model.generate(tokens, max_new_tokens=50)
        text = tokenizer.decode(generated[0][tokens.shape[1]:], skip_special_tokens=True)
        outputs.append((prompt, text))
    return outputs


def measure_diversity(experts_dir: Path, n_layers: int, n_experts: int) -> Dict[str, float]:
    layer_scores: List[float] = []
    sparsities: List[float] = []
    for layer_idx in range(n_layers):
        vectors = []
        for expert_idx in range(n_experts):
            path = experts_dir / f"layer_{layer_idx:02d}_expert_{expert_idx:02d}.safetensors"
            if not path.exists():
                continue
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                gate = handle.get_tensor("gate_ternary").float()
                up = handle.get_tensor("up_ternary").float()
                down = handle.get_tensor("down_ternary").float()
                flat = torch.cat([gate.flatten(), up.flatten(), down.flatten()])
                vectors.append(flat)
                sparsities.append(((flat.abs() < 0.01).float().mean().item()))
        if len(vectors) < 2:
            continue
        cosines = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cos = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)).item()
                cosines.append(cos)
        if cosines:
            layer_scores.append(sum(cosines) / len(cosines))
    return {
        "avg_pairwise_cosine": sum(layer_scores) / len(layer_scores) if layer_scores else 1.0,
        "avg_delta_sparsity": sum(sparsities) / len(sparsities) if sparsities else 1.0,
        "v1_pairwise_cosine": 1.0,
    }


def list_expert_specs(cfg: dict) -> List[ExpertSpec]:
    return [
        ExpertSpec(layer_idx=layer_idx, expert_idx=expert_idx)
        for layer_idx in range(int(cfg["n_layers"]))
        for expert_idx in range(int(cfg["n_experts"]))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential CAKLD expert training for Outlier V3.")
    parser.add_argument("--model-ref", default=DEFAULT_MODEL_REF)
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps-per-expert", type=int, default=EXPERT_STEPS)
    parser.add_argument("--router-steps", type=int, default=ROUTER_STEPS)
    parser.add_argument("--max-experts", type=int, default=None, help="Limit expert count for smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--skip-router", action="store_true")
    parser.add_argument("--skip-checkpoint-write", action="store_true")
    parser.add_argument("--skip-quality-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus_dir = abs_path(args.corpus_dir)
    output_dir = abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = DistillationCorpus(corpus_dir)
    log("=== DISTILLATION CORPUS ===")
    log(f"Examples: {len(corpus)}")
    log(f"Format: {len(corpus.shards)} shards, top_k={corpus.top_k}, max_length={corpus.max_length}")
    log(f"Teacher: {corpus.manifest['teacher']}")

    snapshot_dir = checkpoint_snapshot(args.model_ref)
    cfg = load_config(snapshot_dir)
    budget = print_memory_budget(snapshot_dir, cfg, corpus.max_length, corpus.top_k)
    if budget["total_est"] > 50 * 1024**3:
        raise RuntimeError("Estimated training footprint exceeds 50 GB safety cap; aborting.")
    if args.estimate_only:
        return

    specs = list_expert_specs(cfg)
    if args.max_experts is not None:
        specs = specs[: args.max_experts]

    model, _, _ = build_shared_only_backbone(args.model_ref, args.device)
    device = torch.device(args.device)
    start = time.perf_counter()
    summaries = []
    for idx, spec in enumerate(specs, start=1):
        summary = train_single_expert(
            model,
            corpus,
            spec,
            output_dir=output_dir,
            steps_per_expert=args.steps_per_expert,
            device=device,
        )
        summaries.append(summary)
        if idx == 1 and len(specs) > 1:
            total_minutes = (summary["time_s"] / 60.0) * len(specs)
            log(f"Estimated total expert-training time: {total_minutes/60.0:.1f} hours")

    del model
    gc.collect()
    mps_empty_cache()

    router_summary = None
    if not args.skip_router:
        router_summary = train_router(args.model_ref, corpus, output_dir, args.device, args.router_steps)

    checkpoint_size = 0
    if not args.skip_checkpoint_write:
        write_v3_checkpoint(snapshot_dir, output_dir)
        checkpoint_size = sum(path.stat().st_size for path in output_dir.glob("*.safetensors"))

    quality_outputs: List[Tuple[str, str]] = []
    if not args.skip_quality_check and not args.skip_checkpoint_write:
        quality_outputs = run_quality_check(output_dir, args.device)
        coherent = all(text.strip() for _, text in quality_outputs)
        if not coherent:
            log("WARNING: one or more quick-check generations were empty or incoherent.")

    diversity = measure_diversity(output_dir / "experts", cfg["n_layers"], cfg["n_experts"])
    elapsed = time.perf_counter() - start
    summary = {
        "experts_trained": len(summaries),
        "steps_per_expert": args.steps_per_expert,
        "router_summary": router_summary,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_gb": checkpoint_size / 1024**3,
        "total_time_min": elapsed / 60.0,
        "peak_memory_gb": rss_gb(),
        "diversity": diversity,
    }
    append_jsonl(TRAINING_LOG_PATH, {"phase": "run_complete", **summary})

    log("=== V3 SUMMARY ===")
    log(f"Total experts trained: {len(summaries)}")
    log(f"Checkpoint size: {summary['checkpoint_size_gb']:.2f} GB")
    log(f"Generation time: {summary['total_time_min']:.1f} minutes")
    log(f"Peak memory: {summary['peak_memory_gb']:.2f} GB")
    log(
        f"Expert diversity: V1 cosine=1.0000 -> V3 cosine={diversity['avg_pairwise_cosine']:.4f}; "
        f"avg delta sparsity={diversity['avg_delta_sparsity']:.4f}"
    )
    if router_summary is not None:
        log(
            f"Router tune: loss={router_summary['router_loss']:.4f} "
            f"hit_rate={router_summary['hit_rate']:.3f}"
        )
    if quality_outputs:
        log("=== QUICK QUALITY CHECK ===")
        for prompt, text in quality_outputs:
            log(f"PROMPT: {prompt}")
            log(f"OUTPUT: {text.strip()}\n")


if __name__ == "__main__":
    main()
