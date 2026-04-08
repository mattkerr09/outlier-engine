from __future__ import annotations

import json
import os
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model import OutlierForCausalLM
from .paging import OutlierPagedModel, _default_packed_dir, load_hybrid_paged_qwen
from .tokenizer import OutlierTokenizer, load_tokenizer

_LEGACY_MODEL_ALIASES = {
    # The public README on Outlier-10B describes the V2 dense checkpoint, while
    # the repo contents still point at the older custom MoE artifact. Route the
    # user-facing name to the evaluated V2 model for coherent inference.
    "Outlier-Ai/Outlier-10B": "Outlier-Ai/Outlier-10B-V2",
}


def _canonical_model_ref(model_ref: str) -> str:
    return _LEGACY_MODEL_ALIASES.get(model_ref, model_ref)


def _auto_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _candidate_devices(requested_device: Optional[str]) -> list[str]:
    if requested_device:
        return [requested_device]
    if platform.system() == "Darwin" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return ["mps", "cpu"]
    return [_auto_device()]


def _resolve_model_dir(
    model_ref: str,
    token: Optional[str] = None,
    *,
    canonicalize: bool = True,
) -> Path:
    if canonicalize:
        model_ref = _canonical_model_ref(model_ref)
    path = Path(model_ref).expanduser()
    if path.exists():
        return path.resolve()
    return Path(
        snapshot_download(
            repo_id=model_ref,
            token=token,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.pt",
                "*.py",
                "*.safetensors",
                "*.txt",
                "*.tiktoken",
                "README*",
                "generation_config.json",
                "special_tokens_map.json",
                "tokenizer*",
            ],
        )
    ).resolve()


def _is_local_path(model_ref: str) -> bool:
    return Path(model_ref).expanduser().exists()


def _read_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _normalize_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw)
    field_map = {
        "hidden_size": "hidden_dim",
        "intermediate_size": "intermediate_dim",
        "max_position_embeddings": "max_seq_len",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }
    for source, target in field_map.items():
        if source in cfg and target not in cfg:
            cfg[target] = cfg[source]
    if "outlier_num_experts" in cfg and "n_experts" not in cfg:
        cfg["n_experts"] = cfg["outlier_num_experts"]
    if "outlier_num_experts_per_tok" in cfg and "top_k" not in cfg:
        cfg["top_k"] = cfg["outlier_num_experts_per_tok"]
    return cfg


def _discover_artifacts(model_dir: Path) -> Dict[str, Any]:
    safetensors_files = sorted(model_dir.rglob("*.safetensors"))
    pt_files = sorted(model_dir.rglob("*.pt"))
    expert_files = [path for path in pt_files if path.name.startswith("expert_")]
    return {
        "safetensors_count": len(safetensors_files),
        "pt_count": len(pt_files),
        "expert_pt_count": len(expert_files),
        "sample_safetensors": [str(path.relative_to(model_dir)) for path in safetensors_files[:5]],
        "sample_expert_pts": [str(path.relative_to(model_dir)) for path in expert_files[:5]],
    }


@dataclass
class LoadedOutlier:
    model_ref: str
    model_dir: Path
    config: Dict[str, Any]
    tokenizer: OutlierTokenizer
    model: Any
    device: str
    paged: bool
    artifact_index: Dict[str, Any]
    backend: str = "custom"


def inspect_model(model_ref: str, token: Optional[str] = None) -> Dict[str, Any]:
    resolved_ref = _canonical_model_ref(model_ref)
    if _is_local_path(model_ref):
        model_dir = _resolve_model_dir(model_ref, token=token)
        config = _normalize_config(_read_config(model_dir))
        artifacts = _discover_artifacts(model_dir)
        model_dir_str = str(model_dir)
    else:
        api = HfApi(token=token)
        model_info = api.model_info(resolved_ref)
        config_path = Path(
            hf_hub_download(repo_id=resolved_ref, filename="config.json", token=token)
        )
        config = _normalize_config(json.loads(config_path.read_text(encoding="utf-8")))
        siblings = [sibling.rfilename for sibling in model_info.siblings]
        artifacts = {
            "safetensors_count": sum(1 for name in siblings if name.endswith(".safetensors")),
            "pt_count": sum(1 for name in siblings if name.endswith(".pt")),
            "expert_pt_count": sum(1 for name in siblings if name.endswith(".pt") and "/expert_" in name),
            "sample_safetensors": [name for name in siblings if name.endswith(".safetensors")][:5],
            "sample_expert_pts": [name for name in siblings if name.endswith(".pt") and "/expert_" in name][:5],
        }
        model_dir_str = f"hf://{resolved_ref}"
    return {
        "model_ref": model_ref,
        "resolved_model_ref": resolved_ref,
        "model_dir": model_dir_str,
        "config": config,
        "artifacts": artifacts,
    }


def load_model(
    model_ref: str,
    *,
    token: Optional[str] = None,
    device: Optional[str] = None,
    paged: Optional[bool] = None,
    prefetch: Optional[bool] = None,
    et_routing: Optional[bool] = None,
    max_experts_in_memory: int = 64,
    max_warm_cache: int = 256,
    packed_experts_dir: Optional[str] = None,
) -> LoadedOutlier:
    use_alias = paged is not True
    resolved_ref = _canonical_model_ref(model_ref) if use_alias else model_ref
    model_dir = _resolve_model_dir(model_ref, token=token, canonicalize=use_alias)
    config = _normalize_config(_read_config(model_dir))
    n_experts = int(config.get("n_experts", 0))
    if paged is None:
        paged = config.get("model_type") == "outlier_moe" and n_experts > 0

    tokenizer = load_tokenizer(str(model_dir), token=token)
    last_error: Optional[Exception] = None
    selected_device = _auto_device()
    model = None
    backend = "custom"

    for candidate in _candidate_devices(device):
        try:
            selected_device = candidate
            if paged:
                backend = "hf"
                packed_dir = None
                if packed_experts_dir:
                    packed_dir = packed_experts_dir
                else:
                    default_packed = _default_packed_dir()
                    if default_packed.joinpath("index.json").exists():
                        packed_dir = str(default_packed)
                    else:
                        warnings.warn(
                            f"No packed expert cache found at {default_packed}. "
                            f"Run `outlier-engine repack {model_ref}` first for faster paged loads.",
                            stacklevel=2,
                        )
                try:
                    model = load_hybrid_paged_qwen(
                        str(model_dir),
                        device=candidate,
                        max_experts_in_memory=max_experts_in_memory,
                        max_warm_cache=max_warm_cache,
                        packed_experts_dir=packed_dir,
                    )
                except Exception as hybrid_exc:
                    warnings.warn(f"Hybrid paged loader failed, falling back to legacy paged runtime: {hybrid_exc}")
                    backend = "paged"
                    model = OutlierPagedModel(
                        str(model_dir),
                        device=candidate,
                        max_experts_in_memory=max_experts_in_memory,
                        max_warm_cache=max_warm_cache,
                        packed_experts_dir=packed_dir,
                    )
            else:
                if config.get("model_type") == "outlier_moe":
                    backend = "custom"
                    model = OutlierForCausalLM.load_from_pretrained(str(model_dir), token=token)
                    if candidate != "cpu":
                        model = model.to(candidate)
                    model.eval()
                else:
                    from transformers import AutoModelForCausalLM

                    backend = "hf"
                    dtype = torch.float16 if candidate == "mps" else "auto"
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_dir),
                        trust_remote_code=True,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                    )
                    model = model.to(candidate)
                    model.eval()
            break
        except Exception as exc:
            last_error = exc
            model = None
            if device is not None or candidate == "cpu":
                raise
            warnings.warn(f"Falling back from {candidate} to cpu: {exc}")

    if model is None:
        assert last_error is not None
        raise last_error

    prefetch_enabled = prefetch
    if prefetch_enabled is None:
        prefetch_enabled = os.environ.get("OUTLIER_PREFETCH", "").strip() == "1"
    if prefetch_enabled and paged and hasattr(model, "enable_expert_prefetch"):
        model.enable_expert_prefetch()

    et_enabled = et_routing
    if et_enabled is None:
        et_enabled = os.environ.get("OUTLIER_ET_ROUTING", "").strip() == "1"
    if et_enabled and paged and hasattr(model, "enable_et_routing"):
        model.enable_et_routing()

    return LoadedOutlier(
        model_ref=resolved_ref,
        model_dir=model_dir,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=selected_device,
        paged=bool(paged),
        artifact_index=_discover_artifacts(model_dir),
        backend=backend,
    )
