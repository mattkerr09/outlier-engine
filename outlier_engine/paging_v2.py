"""
Paging v2: monolith-backed expert loading integrated with the production paged
inference path.

This module keeps the proven batched-GEMM decode path and cross-layer prefetch
logic from `paging.py`, but swaps expert reads onto the single-file monolith
format from `experiments/monolith`.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from types import MethodType
from typing import Dict, Optional

import numpy as np
import torch

from .expert_store import ExpertStore, SUB_FILE_ORDER
from .paging import (
    ExpertPageManager,
    OutlierPagedModel,
    _HybridPagedMLP,
    _normalize_config,
    _ExpertWeights,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_monolith_path(
    model_dir: Path,
    packed_experts_dir: Optional[Path],
    monolith_path: Optional[str | Path],
) -> Optional[Path]:
    candidates: list[Path] = []
    if monolith_path:
        candidates.append(Path(monolith_path).expanduser())
    env_path = os.environ.get("OUTLIER_MONOLITH_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    if packed_experts_dir is not None:
        candidates.append(packed_experts_dir / "experts.bin")
    candidates.extend(
        [
            model_dir / "packed_experts" / "experts.bin",
            _repo_root() / "experiments" / "monolith" / "experts.bin",
            Path.home() / "outlier-engine" / "packed_experts" / "experts.bin",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


class MonolithExpertLoader:
    """Persistent random-access reader for packed experts.bin monoliths."""

    def __init__(self, store_path: str | Path, packed_index: Dict[str, Dict[str, object]]) -> None:
        self.store_path = Path(store_path).expanduser().resolve()
        self._packed_index = packed_index
        self._fd = os.open(str(self.store_path), os.O_RDONLY)
        with open(self.store_path, "rb") as handle:
            header = ExpertStore._read_header(handle)
            self._index = ExpertStore._read_index(handle, header["num_entries"])
        self._sub_sizes = {
            name: int(size)
            for name, size in zip(SUB_FILE_ORDER, header["sub_sizes"])
        }

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _meta(self, layer_idx: int, expert_idx: int, proj: str, suffix: str) -> Dict[str, object]:
        key = f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj}_{suffix}"
        info = self._packed_index.get(key)
        if info is None:
            raise KeyError(f"Packed metadata missing for {key}")
        return info

    def load_expert(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        key = (layer_idx, expert_idx)
        if key not in self._index:
            raise KeyError(f"Expert ({layer_idx}, {expert_idx}) not present in monolith store")

        offset, size = self._index[key]
        raw = os.pread(self._fd, size, offset)
        if len(raw) != size:
            raise IOError(
                f"Short read for expert ({layer_idx}, {expert_idx}): expected {size} bytes, got {len(raw)}"
            )

        view = memoryview(raw)
        cursor = 0
        chunks: dict[str, memoryview] = {}
        for name in SUB_FILE_ORDER:
            chunk_size = self._sub_sizes[name]
            chunks[name] = view[cursor : cursor + chunk_size]
            cursor += chunk_size

        tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for proj in ("gate", "up", "down"):
            ternary_info = self._meta(layer_idx, expert_idx, proj, "ternary")
            scale_info = self._meta(layer_idx, expert_idx, proj, "scale")
            ternary = torch.from_numpy(
                np.frombuffer(chunks[f"{proj}_ternary"], dtype=np.uint8).copy()
            )
            scale_shape = tuple(int(dim) for dim in scale_info["shape"])
            scale_arr = np.frombuffer(chunks[f"{proj}_scale"], dtype=np.float16).copy()
            scale = torch.from_numpy(scale_arr).reshape(scale_shape)
            tensors[proj] = (ternary, scale)
            shapes[proj] = tuple(int(dim) for dim in ternary_info["shape"])

        return _ExpertWeights(
            tensors["gate"][0],
            tensors["gate"][1],
            tensors["up"][0],
            tensors["up"][1],
            tensors["down"][0],
            tensors["down"][1],
            packed=True,
            dequantized=False,
            packed_format="tq10",
            shapes=shapes,
        )


class _MonolithMixin:
    def _init_monolith_loader(
        self,
        *,
        model_dir: Path,
        packed_experts_dir: Optional[Path],
        monolith_path: Optional[str | Path],
    ) -> None:
        self._monolith_loader: MonolithExpertLoader | None = None
        self._monolith_path: Path | None = None

        if not getattr(self, "_packed_index", None):
            warnings.warn(
                "paging_v2 requires packed expert metadata; falling back to paging.py disk loads.",
                stacklevel=2,
            )
            return

        resolved = _resolve_monolith_path(model_dir, packed_experts_dir, monolith_path)
        if resolved is None:
            warnings.warn(
                "No experts.bin monolith found for paging_v2; falling back to paging.py disk loads.",
                stacklevel=2,
            )
            return

        self._monolith_path = resolved
        self._monolith_loader = MonolithExpertLoader(resolved, self._packed_index)

    def _load_expert_from_disk(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        loader = getattr(self, "_monolith_loader", None)
        if loader is not None:
            return loader.load_expert(layer_idx, expert_idx)
        return super()._load_expert_from_disk(layer_idx, expert_idx)


class ExpertPageManagerV2(_MonolithMixin, ExpertPageManager):
    """Hybrid page manager that loads packed experts from the monolith store."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "mps",
        *,
        n_experts: int,
        n_layers: int = 0,
        top_k: int = 2,
        max_experts_in_memory: int = 64,
        max_warm_cache: int = 256,
        packed_experts_dir: Optional[str] = None,
        monolith_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            model_dir,
            device=device,
            n_experts=n_experts,
            n_layers=n_layers,
            top_k=top_k,
            max_experts_in_memory=max_experts_in_memory,
            max_warm_cache=max_warm_cache,
            packed_experts_dir=packed_experts_dir,
        )
        self._init_monolith_loader(
            model_dir=self.model_dir,
            packed_experts_dir=self._packed_experts_dir,
            monolith_path=monolith_path,
        )


class OutlierPagedModelV2(_MonolithMixin, OutlierPagedModel):
    """Legacy paged runtime with monolith-backed expert loads."""

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        max_experts_in_memory: int = 4,
        max_warm_cache: int = 256,
        packed_experts_dir: Optional[str] = None,
        monolith_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(
            model_path,
            device=device,
            max_experts_in_memory=max_experts_in_memory,
            max_warm_cache=max_warm_cache,
            packed_experts_dir=packed_experts_dir,
        )
        self._init_monolith_loader(
            model_dir=Path(model_path).expanduser().resolve(),
            packed_experts_dir=self._packed_experts_dir,
            monolith_path=monolith_path,
        )


def load_hybrid_paged_qwen_v2(
    model_path: str | Path,
    *,
    device: str = "mps",
    max_experts_in_memory: int = 64,
    max_warm_cache: int = 256,
    packed_experts_dir: Optional[str] = None,
    monolith_path: Optional[str | Path] = None,
):
    """Build the hybrid Qwen paged runtime backed by experts.bin monolith reads."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model_dir = Path(model_path)
    with open(model_dir / "config.json", encoding="utf-8") as handle:
        cfg = _normalize_config(json.load(handle))

    hf_cfg = Qwen2Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_dim"],
        intermediate_size=cfg["intermediate_dim"],
        num_hidden_layers=cfg["n_layers"],
        num_attention_heads=cfg["n_heads"],
        num_key_value_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
        hidden_act="silu",
        max_position_embeddings=cfg.get("max_seq_len", 32768),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        use_cache=True,
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        rope_parameters=cfg.get(
            "rope_parameters",
            {"rope_theta": cfg.get("rope_theta", 1000000.0), "rope_type": "default"},
        ),
        use_sliding_window=cfg.get("use_sliding_window", False),
        sliding_window=cfg.get("sliding_window"),
        max_window_layers=cfg.get("max_window_layers", cfg["n_layers"]),
        layer_types=cfg.get("layer_types"),
        attention_dropout=cfg.get("attention_dropout", 0.0),
        bos_token_id=cfg.get("bos_token_id"),
        eos_token_id=cfg.get("eos_token_id"),
        pad_token_id=cfg.get("pad_token_id"),
    )

    with torch.device("meta"):
        model = Qwen2ForCausalLM(hf_cfg)
        for layer_idx, layer in enumerate(model.model.layers):
            layer.mlp = _HybridPagedMLP(
                cfg["hidden_dim"],
                cfg["intermediate_dim"],
                cfg.get("n_experts", 0),
                cfg.get("top_k", 2),
                layer_idx=layer_idx,
                page_manager=None,
            )

    model = model.to_empty(device=torch.device(device))
    model = model.to(dtype=torch.bfloat16)

    page_manager = ExpertPageManagerV2(
        model_dir,
        device=device,
        n_experts=cfg.get("n_experts", 0),
        n_layers=cfg["n_layers"],
        top_k=cfg.get("top_k", 2),
        max_experts_in_memory=max_experts_in_memory,
        max_warm_cache=max_warm_cache,
        packed_experts_dir=packed_experts_dir,
        monolith_path=monolith_path,
    )

    for layer in model.model.layers:
        if isinstance(layer.mlp, _HybridPagedMLP):
            layer.mlp.page_manager = page_manager

    from safetensors import safe_open
    from .quantize_utils import quantize_to_int8

    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for raw_key in handle.keys():
                if raw_key.startswith("base.model.layers.") and ".mlp.experts." in raw_key:
                    continue
                tensor = handle.get_tensor(raw_key)
                if raw_key.startswith("base.model."):
                    key = raw_key[len("base.") :]
                elif raw_key.startswith("base."):
                    key = raw_key[len("base.") :]
                else:
                    key = raw_key

                if key == "model.embed_tokens.weight":
                    model.model.embed_tokens.weight.data.copy_(tensor.to(model.model.embed_tokens.weight.dtype))
                    continue
                if key == "model.norm.weight":
                    model.model.norm.weight.data.copy_(tensor.to(model.model.norm.weight.dtype))
                    continue
                if key == "lm_head.weight":
                    model.lm_head.weight.data.copy_(tensor.to(model.lm_head.weight.dtype))
                    continue

                layer_match = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
                if not layer_match:
                    continue
                layer_idx = int(layer_match.group(1))
                suffix = layer_match.group(2)
                layer = model.model.layers[layer_idx]

                if suffix == "input_layernorm.weight":
                    layer.input_layernorm.weight.data.copy_(tensor.to(layer.input_layernorm.weight.dtype))
                elif suffix == "post_attention_layernorm.weight":
                    layer.post_attention_layernorm.weight.data.copy_(tensor.to(layer.post_attention_layernorm.weight.dtype))
                elif suffix.startswith("self_attn."):
                    proj_name, param_name = suffix[len("self_attn.") :].rsplit(".", 1)
                    module = getattr(layer.self_attn, proj_name, None)
                    target = getattr(module, param_name, None) if module is not None else None
                    if target is not None:
                        target.data.copy_(tensor.to(target.dtype))
                elif suffix == "mlp.router.weight" and isinstance(layer.mlp, _HybridPagedMLP):
                    layer.mlp.router_weight.data.copy_(tensor.to(layer.mlp.router_weight.dtype))
                elif suffix.startswith("mlp.shared_expert.") and isinstance(layer.mlp, _HybridPagedMLP):
                    shared = layer.mlp.shared
                    proj = suffix[len("mlp.shared_expert.") :]
                    if proj == "gate_W":
                        q, s = quantize_to_int8(tensor)
                        shared.gate_w.copy_(q)
                        shared.gate_s.copy_(s.to(shared.gate_s.dtype))
                    elif proj == "up_W":
                        q, s = quantize_to_int8(tensor)
                        shared.up_w.copy_(q)
                        shared.up_s.copy_(s.to(shared.up_s.dtype))
                    elif proj == "down_W":
                        q, s = quantize_to_int8(tensor)
                        shared.down_w.copy_(q)
                        shared.down_s.copy_(s.to(shared.down_s.dtype))

    model.eval()
    model.outlier_device = torch.device(device)
    model.outlier_page_manager = page_manager
    model.cache_stats = MethodType(lambda self: self.outlier_page_manager.cache_stats(), model)
    model.prefetch_stats = MethodType(lambda self: self.outlier_page_manager.prefetch_stats(), model)

    def _enable_expert_prefetch(self) -> None:
        self.outlier_page_manager.enable_expert_prefetch()
        if getattr(self, "_outlier_prefetch_hooks", None):
            return

        handles = []
        for layer_idx, layer in enumerate(self.model.layers):
            def _wait_hook(_module, _inputs, current_layer=layer_idx):
                self.outlier_page_manager.wait_for_layer(current_layer)

            handles.append(layer.register_forward_pre_hook(_wait_hook))
        self._outlier_prefetch_hooks = handles

    model.enable_expert_prefetch = MethodType(_enable_expert_prefetch, model)

    def _enable_et_routing(self) -> None:
        self.outlier_page_manager.enable_et_routing()

    model.enable_et_routing = MethodType(_enable_et_routing, model)
    model.routing_stats = MethodType(lambda self: self.outlier_page_manager.et_routing_stats(), model)
    return model


__all__ = [
    "ExpertPageManagerV2",
    "MonolithExpertLoader",
    "OutlierPagedModelV2",
    "load_hybrid_paged_qwen_v2",
]
