"""Outlier MoE modeling file with GPUResidentExpert dequant port.

Ported from Outlier-150B-V3.2/modeling_outlier_150b_rexmoe.py.
Key change: one-time dequant of all experts to GPU at load time,
eliminating the per-token CPU materialize() bottleneck that makes
inference 56× slower.
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, PreTrainedModel

from .configuration_outlier_moe import OutlierMoEConfig


def _parse_dtype(value):
    if value is None or value == "auto":
        return value
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        table = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        return table.get(value.lower(), value)
    return value


def _load_alpha_map(model_dir: Path) -> dict[int, dict[int, float]]:
    path = model_dir / "alpha.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[int, float]] = {}
    for key, value in raw.items():
        match = re.match(r"layer_(\d+)_expert_(\d+)", key)
        if not match:
            continue
        layer_idx = int(match.group(1))
        expert_idx = int(match.group(2))
        out.setdefault(layer_idx, {})[expert_idx] = float(value)
    return out


def _load_router_map(model_dir: Path) -> dict[int, torch.Tensor]:
    path = model_dir / "router_state.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing router state: {path}")
    out: dict[int, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            match = re.match(r"layer_(\d+)_router_weight", key)
            if match:
                out[int(match.group(1))] = handle.get_tensor(key).float()
    if not out:
        raise RuntimeError(f"No router weights found in {path}")
    return out


class CPUQuantizedExpert:
    """Expert weights stored on CPU in ternary int8. Used during loading only."""
    def __init__(self, tensors: dict[str, torch.Tensor], alpha: float) -> None:
        self.gate_ternary = tensors["gate_ternary"].to(torch.int8).cpu()
        self.gate_scale = tensors["gate_scale"].to(torch.float16).cpu()
        self.up_ternary = tensors["up_ternary"].to(torch.int8).cpu()
        self.up_scale = tensors["up_scale"].to(torch.float16).cpu()
        self.down_ternary = tensors["down_ternary"].to(torch.int8).cpu()
        self.down_scale = tensors["down_scale"].to(torch.float16).cpu()
        self.alpha = float(alpha)
        self.effective_alpha = float(alpha)

    def dequantize(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-time dequant to GPU-resident dense weights."""
        gate = self.gate_ternary.to(device=device, dtype=dtype) * self.gate_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        up = self.up_ternary.to(device=device, dtype=dtype) * self.up_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        down = self.down_ternary.to(device=device, dtype=dtype) * self.down_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        return gate, up, down


class GPUResidentExpert:
    """Expert delta weights dequantized once and kept resident on GPU.

    Ported from Outlier-150B-V3.2/modeling_outlier_150b_rexmoe.py line 49.
    This is the key optimization: no per-token CPU materialize().
    """
    __slots__ = ("gate_w", "up_w", "down_w", "effective_alpha")
    def __init__(self, gate_w, up_w, down_w, effective_alpha):
        self.gate_w = gate_w
        self.up_w = up_w
        self.down_w = down_w
        self.effective_alpha = effective_alpha

    def dequantize(self, device, dtype):
        return self.gate_w, self.up_w, self.down_w


class EvalRoutedQuantizedMoE(nn.Module):
    def __init__(self, shared_mlp: nn.Module, experts: dict, router_weight: torch.Tensor, *, top_k: int) -> None:
        super().__init__()
        self.shared_mlp = shared_mlp
        self.experts = experts
        self.register_buffer("router_weight", router_weight.float().contiguous(), persistent=False)
        self.top_k = int(top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_mlp(x)
        batch, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        shared_flat = shared_out.reshape(-1, hidden)
        router_weight = self.router_weight.to(device=x.device, dtype=torch.float32)
        logits = F.linear(x_flat.float(), router_weight)
        vals, idx = torch.topk(logits, k=min(self.top_k, router_weight.shape[0]), dim=-1)
        weights = F.softmax(vals, dim=-1)
        mixed = shared_flat.float()
        target_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float32

        for expert_idx, expert in self.experts.items():
            token_idx, choice_idx = torch.where(idx == expert_idx)
            if token_idx.numel() == 0:
                continue
            gate_w, up_w, down_w = expert.dequantize(x.device, target_dtype)
            # Expert weights may be on a different GPU (round-robin placement)
            # Move input tokens to the expert's device for matmul
            expert_dev = gate_w.device
            x_tok = x_flat[token_idx].to(device=expert_dev, dtype=target_dtype)
            gate = F.linear(x_tok, gate_w)
            up = F.linear(x_tok, up_w)
            out = F.linear(F.silu(gate) * up, down_w)
            # Move result back to the mixed tensor's device (cuda:0)
            out = out.to(device=mixed.device)
            delta = out.float() - shared_flat[token_idx].float()
            mixed[token_idx] += weights[token_idx, choice_idx].unsqueeze(-1) * expert.effective_alpha * delta

        return mixed.to(dtype=shared_out.dtype).reshape(batch, seq_len, hidden)


def _load_layer_experts(model_dir: Path, layer_idx: int, experts_per_layer: int, alpha_map: dict[int, dict[int, float]]) -> dict[int, CPUQuantizedExpert]:
    expert_dir = model_dir / "experts"
    layer_alphas = alpha_map.get(layer_idx, {})
    experts: dict[int, CPUQuantizedExpert] = {}
    for expert_idx in range(experts_per_layer):
        path = expert_dir / f"layer_{layer_idx:02d}_expert_{expert_idx:02d}.safetensors"
        if not path.exists():
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        experts[expert_idx] = CPUQuantizedExpert(tensors, layer_alphas.get(expert_idx, 0.0))
    return experts


class OutlierMoEForCausalLM(PreTrainedModel):
    config_class = OutlierMoEConfig

    def __init__(self, config: OutlierMoEConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        model_dir = Path(pretrained_model_name_or_path)
        if config is None:
            config = OutlierMoEConfig.from_pretrained(model_dir)

        base_kwargs = {}
        for key in ("trust_remote_code", "device_map", "low_cpu_mem_usage", "attn_implementation"):
            if key in kwargs:
                base_kwargs[key] = kwargs.pop(key)

        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is None and "dtype" in kwargs:
            torch_dtype = kwargs.pop("dtype")
        if torch_dtype is not None:
            base_kwargs["torch_dtype"] = _parse_dtype(torch_dtype)

        # Force multi-GPU split to leave room for GPU-resident experts.
        # Cap per-GPU memory so device_map="auto" distributes transformer
        # layers across GPUs, then experts follow their layer's device.
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and "device_map" not in base_kwargs:
            base_kwargs["device_map"] = "auto"
            # Cap at 80 GiB per GPU to force even split of transformer layers,
            # leaving ~100 GiB per GPU for expert dequant on each side
            base_kwargs["max_memory"] = {i: "80GiB" for i in range(n_gpus)}
            print(f"Forcing {n_gpus}-GPU split (max 80GiB each, reserves room for experts)", flush=True)
        elif "device_map" not in base_kwargs:
            base_kwargs["device_map"] = {"": 0}  # single GPU

        print(f"Loading base model from {config.base_model_name_or_path} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            **base_kwargs,
        )

        alpha_map = _load_alpha_map(model_dir)
        router_map = _load_router_map(model_dir)
        layers = list(getattr(config, "moe_layers", []))
        experts_per_layer = int(getattr(config, "n_experts", 0))
        top_k = int(getattr(config, "top_k", 2))

        print(f"Loading experts for {len(layers)} MoE layers...", flush=True)
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            experts = _load_layer_experts(model_dir, layer_idx, experts_per_layer, alpha_map)
            router_weight = router_map[layer_idx]
            layer.mlp = EvalRoutedQuantizedMoE(layer.mlp, experts, router_weight, top_k=top_k)

        # ── GPU-resident expert materialization (ported from 150B) ──
        # One-time dequant of all experts, distributed across GPUs.
        # Round-robin MoE layers across GPUs to balance VRAM usage.
        print("Materializing experts onto GPU (one-time dequant)...", flush=True)
        dequant_cache = {}
        n_materialized = 0

        for li, layer_idx in enumerate(layers):
            layer = model.model.layers[layer_idx]
            moe = layer.mlp
            if not isinstance(moe, EvalRoutedQuantizedMoE):
                continue

            # Round-robin expert placement across GPUs for even distribution
            if n_gpus > 1:
                target_gpu = li % n_gpus
                dev = torch.device(f"cuda:{target_gpu}")
            else:
                dev = next(moe.shared_mlp.parameters()).device
            dtype = torch.bfloat16

            new_experts = {}
            for ei, expert in moe.experts.items():
                dk = (layer_idx, ei, str(dev))
                if dk not in dequant_cache:
                    g, u, d = expert.dequantize(dev, dtype)
                    dequant_cache[dk] = (g.contiguous(), u.contiguous(), d.contiguous())
                    if n_materialized % 20 == 0:
                        mem0 = torch.cuda.memory_allocated(0) / 1024**3
                        mem1 = torch.cuda.memory_allocated(1) / 1024**3 if n_gpus > 1 else 0
                        print(f"    layer {layer_idx} expert {ei} → {dev} "
                              f"(GPU0: {mem0:.1f}GB, GPU1: {mem1:.1f}GB)", flush=True)
                g, u, d = dequant_cache[dk]
                new_experts[ei] = GPUResidentExpert(g, u, d, expert.effective_alpha)
            moe.experts = new_experts
            n_materialized += 1

        # Free CPU expert data
        print(f"  Materialized {n_materialized}/{len(layers)} layers, {len(dequant_cache)} unique expert dequants", flush=True)
        del dequant_cache
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(torch.cuda.device_count()):
            print(f"  GPU{i}: {torch.cuda.memory_allocated(i)/1024**3:.1f}GB", flush=True)

        model.config = config
        print("Model ready.", flush=True)
        return model
