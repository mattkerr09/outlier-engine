from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from .model import OutlierForCausalLM
from .paging import OutlierPagedModel
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
    # The PyTorch MPS fallback path is currently much slower and harder to
    # debug than the CPU path for this runtime, so prefer CPU on macOS.
    if platform.system() == "Darwin":
        return "cpu"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_model_dir(model_ref: str, token: Optional[str] = None) -> Path:
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
    max_experts_in_memory: int = 4,
    max_warm_cache: int = 16,
) -> LoadedOutlier:
    resolved_ref = _canonical_model_ref(model_ref)
    model_dir = _resolve_model_dir(model_ref, token=token)
    config = _normalize_config(_read_config(model_dir))
    device = device or _auto_device()
    n_experts = int(config.get("n_experts", 0))
    if paged is None:
        paged = config.get("model_type") == "outlier_moe" and n_experts > 0

    tokenizer = load_tokenizer(str(model_dir), token=token)
    if paged:
        model = OutlierPagedModel(
            str(model_dir),
            device=device,
            max_experts_in_memory=max_experts_in_memory,
            max_warm_cache=max_warm_cache,
        )
    else:
        if config.get("model_type") == "outlier_moe":
            model = OutlierForCausalLM.load_from_pretrained(str(model_dir), token=token)
            if device != "cpu":
                model = model.to(device)
            model.eval()
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                torch_dtype="auto",
            )
            if device != "cpu":
                model = model.to(device)
            model.eval()

    return LoadedOutlier(
        model_ref=resolved_ref,
        model_dir=model_dir,
        config=config,
        tokenizer=tokenizer,
        model=model,
        device=device,
        paged=bool(paged),
        artifact_index=_discover_artifacts(model_dir),
    )
