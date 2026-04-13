from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, PreTrainedModel

from .configuration_outlier_150b_rexmoe import OutlierReXMoEConfig


def _parse_dtype(value):
    if value is None or value == "auto":
        return value
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        table = {
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16,   "fp16": torch.float16,
            "float32": torch.float32,   "fp32": torch.float32,
        }
        return table.get(value.lower(), value)
    return value


class CPUQuantizedExpert:
    """Holds ternary-quantized expert delta weights on CPU."""
    def __init__(self, tensors: dict[str, torch.Tensor], alpha: float, psr_scale: float) -> None:
        self.gate_ternary = tensors["gate_ternary"].to(torch.int8).cpu()
        self.gate_scale   = tensors["gate_scale"].to(torch.float16).cpu()
        self.up_ternary   = tensors["up_ternary"].to(torch.int8).cpu()
        self.up_scale     = tensors["up_scale"].to(torch.float16).cpu()
        self.down_ternary = tensors["down_ternary"].to(torch.int8).cpu()
        self.down_scale   = tensors["down_scale"].to(torch.float16).cpu()
        self.effective_alpha = float(alpha) * float(psr_scale)

    def dequantize(self, device: torch.device, dtype: torch.dtype):
        gate = self.gate_ternary.to(device=device, dtype=dtype) * self.gate_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        up   = self.up_ternary.to(device=device, dtype=dtype)   * self.up_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        down = self.down_ternary.to(device=device, dtype=dtype) * self.down_scale.to(device=device, dtype=dtype).unsqueeze(-1)
        return gate, up, down


class GPUResidentExpert:
    """Expert delta weights dequantized once and kept resident on GPU."""
    __slots__ = ("gate_w", "up_w", "down_w", "effective_alpha")
    def __init__(self, gate_w, up_w, down_w, effective_alpha):
        self.gate_w = gate_w
        self.up_w = up_w
        self.down_w = down_w
        self.effective_alpha = effective_alpha

    def dequantize(self, device, dtype):
        return self.gate_w, self.up_w, self.down_w


class ReXMoEInferenceMLP(nn.Module):
    """Inference-time ReXMoE MLP for a single layer."""

    def __init__(
        self,
        base_mlp: nn.Module,
        experts: dict,
        router_weight: torch.Tensor,
        *,
        top_k: int,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.experts = experts
        self.register_buffer("router_weight", router_weight.float().contiguous(), persistent=False)
        self.top_k = int(top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_mlp(x)
        batch, seq_len, hidden = x.shape
        x_flat  = x.reshape(-1, hidden)
        bo_flat = base_out.reshape(-1, hidden)

        router_w = self.router_weight.to(device=x.device, dtype=torch.float32)
        logits   = F.linear(x_flat.float(), router_w)
        vals, idx = torch.topk(logits, k=min(self.top_k, router_w.shape[0]), dim=-1)
        weights  = F.softmax(vals, dim=-1)

        mixed = bo_flat.float()
        target_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

        for expert_idx, expert in self.experts.items():
            token_idx, choice_idx = torch.where(idx == expert_idx)
            if token_idx.numel() == 0:
                continue
            gate_w, up_w, down_w = expert.dequantize(x.device, target_dtype)
            x_tok = x_flat[token_idx].to(dtype=target_dtype)
            gate  = F.linear(x_tok, gate_w)
            up    = F.linear(x_tok, up_w)
            out   = F.linear(F.silu(gate) * up, down_w)
            delta = out.float() - bo_flat[token_idx].float()
            mixed[token_idx] += (
                weights[token_idx, choice_idx].unsqueeze(-1)
                * expert.effective_alpha
                * delta
            )
            del x_tok, gate, up, out, delta

        return mixed.to(dtype=base_out.dtype).reshape(batch, seq_len, hidden)


def _load_group_experts(
    model_dir: Path,
    group_idx: int,
    n_experts: int,
    alpha_map: dict[str, float],
    psr_scale: float,
) -> dict[int, CPUQuantizedExpert]:
    path = model_dir / f"experts_grp_{group_idx:02d}.safetensors"
    experts: dict[int, CPUQuantizedExpert] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for ei in range(n_experts):
            tensors = {}
            for key in ("gate_ternary", "gate_scale", "up_ternary", "up_scale", "down_ternary", "down_scale"):
                full_key = f"expert_{ei:02d}_{key}"
                if full_key in handle.keys():
                    tensors[key] = handle.get_tensor(full_key)
            if tensors:
                alpha_key = f"group_{group_idx:02d}_expert_{ei:02d}"
                alpha = float(alpha_map.get(alpha_key, 0.0))
                experts[ei] = CPUQuantizedExpert(tensors, alpha, psr_scale)
    return experts


class OutlierReXMoEForCausalLM(PreTrainedModel):
    config_class = OutlierReXMoEConfig

    def __init__(self, config: OutlierReXMoEConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        model_dir = Path(pretrained_model_name_or_path)
        if config is None:
            config = OutlierReXMoEConfig.from_pretrained(model_dir)

        base_kwargs = {}
        for key in ("trust_remote_code", "device_map", "low_cpu_mem_usage", "attn_implementation"):
            if key in kwargs:
                base_kwargs[key] = kwargs.pop(key)

        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is None and "dtype" in kwargs:
            torch_dtype = kwargs.pop("dtype")
        if torch_dtype is not None:
            base_kwargs["torch_dtype"] = _parse_dtype(torch_dtype)

        # Force multi-GPU split to leave room for GPU-resident experts
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            base_kwargs["device_map"] = "auto"
            base_kwargs["max_memory"] = {i: "75GiB" for i in range(n_gpus)}
            print(f"Forcing {n_gpus}-GPU split (max 75GiB each) for expert residency", flush=True)

        print(f"Loading base model from {config.base_model_name_or_path} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            **base_kwargs,
        )

        # Load metadata
        alpha_map  = json.loads((model_dir / "alpha.json").read_text())
        group_map  = json.loads((model_dir / "group_map.json").read_text())

        # Load all router weights
        from safetensors.torch import load_file
        router_tensors = load_file(str(model_dir / "router_state.safetensors"))

        # Cache group experts per (group_idx, psr_scale) to avoid reloading
        group_expert_cache: dict[tuple[int, float], dict[int, CPUQuantizedExpert]] = {}

        print(f"Patching {len(config.moe_layers)} MoE layers...", flush=True)
        for layer_idx in config.moe_layers:
            info       = group_map.get(str(layer_idx), {})
            group_idx  = info.get("group_idx", 1)
            psr_scale  = info.get("psr_scale", 1.0)
            cache_key  = (group_idx, psr_scale)

            if cache_key not in group_expert_cache:
                group_expert_cache[cache_key] = _load_group_experts(
                    model_dir, group_idx, config.n_experts, alpha_map, psr_scale
                )
                print(f"  Loaded group {group_idx:02d} experts (psr={psr_scale})", flush=True)

            experts = group_expert_cache[cache_key]
            router_w = router_tensors[f"layer_{layer_idx}_router_weight"]

            layer = model.model.layers[layer_idx]
            layer.mlp = ReXMoEInferenceMLP(
                layer.mlp, experts, router_w, top_k=config.top_k
            )

        # ── GPU-resident expert materialization ──────────────────────────────
        # One-time dequant of all experts onto the GPU where their layer lives.
        # Shared across layers in the same group on the same device.
        print("Materializing experts onto GPU (one-time dequant)...", flush=True)
        dequant_cache = {}  # (group_idx, expert_idx, device_str) -> (gate, up, down)
        n_materialized = 0

        for layer_idx in config.moe_layers:
            layer = model.model.layers[layer_idx]
            moe = layer.mlp
            if not isinstance(moe, ReXMoEInferenceMLP):
                print(f"  WARN: layer {layer_idx} mlp is {type(moe).__name__}, not ReXMoEInferenceMLP", flush=True)
                continue
            dev = next(moe.base_mlp.parameters()).device
            dtype = next(moe.base_mlp.parameters()).dtype
            info = group_map.get(str(layer_idx), {})
            gi = info.get("group_idx", 1)

            new_experts = {}
            for ei, expert in moe.experts.items():
                dk = (gi, ei, str(dev))
                if dk not in dequant_cache:
                    g, u, d = expert.dequantize(dev, dtype)
                    dequant_cache[dk] = (g.contiguous(), u.contiguous(), d.contiguous())
                g, u, d = dequant_cache[dk]
                new_experts[ei] = GPUResidentExpert(g, u, d, expert.effective_alpha)
            moe.experts = new_experts
            n_materialized += 1

        # Free CPU expert data
        print(f"  Materialized {n_materialized}/{len(config.moe_layers)} layers, {len(dequant_cache)} unique expert dequants", flush=True)
        del group_expert_cache, dequant_cache
        gc.collect()
        torch.cuda.empty_cache()

        mem_parts = []
        for i in range(torch.cuda.device_count()):
            mem_parts.append(f"GPU{i}:{torch.cuda.memory_allocated(i)/1024**3:.1f}GB")
        print("  Experts resident. " + " ".join(mem_parts), flush=True)

        model.config = config
        print("Model ready.", flush=True)
        return model
