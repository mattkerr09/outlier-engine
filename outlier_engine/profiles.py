"""
Alpha profile management for Outlier MoE domain mode switching.

OUTLIER-LOCAL-EXPERIMENTS-001 / Experiment 2: Domain Mode Switching

An alpha profile is a JSON file containing per-layer, per-expert alpha
scalars.  Loading a profile overwrites the in-memory alphas of every
_HybridPagedMLP layer — takes <1 ms (just reassigning floats).

File format:
  {
    "<layer_idx>": {"<expert_idx>": <alpha_float>, ...},
    ...
    "__meta__": {"n_experts": N, "n_layers": M, ...}
  }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paging import _HybridPagedMLP
from .ttt import (
    _get_moe_layers,
    collect_alpha_state,
    _tokenize,
    _generate_ids,
    _MEDICAL_TEXT,
    _CODING_TEXT,
    _LEGAL_TEXT,
    check_ram,
    free_ram_gb,
)


# ---------------------------------------------------------------------------
# Core profile I/O
# ---------------------------------------------------------------------------

def save_alpha_profile(model, path: str, *, label: str = "") -> None:
    """Serialize current alpha values from model to a JSON file.

    Args:
        model:  The Qwen2ForCausalLM with _HybridPagedMLP layers.
        path:   Destination .json file path (created or overwritten).
        label:  Optional human-readable label stored in __meta__.
    """
    moe_layers = _get_moe_layers(model)
    profile: Dict[str, Any] = {}
    for layer_idx, mlp in moe_layers:
        profile[str(layer_idx)] = {
            str(e): mlp.alphas.get(e, mlp.alpha_default)
            for e in range(mlp.n_experts)
        }

    n_experts = moe_layers[0][1].n_experts if moe_layers else 0
    profile["__meta__"] = {
        "n_moe_layers": len(moe_layers),
        "n_experts": n_experts,
        "label": label,
    }

    Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_alpha_profile(model, path: str) -> float:
    """Load a JSON alpha profile and apply it to model in-place.

    Returns the wall-clock time in milliseconds for the switch.

    Only layers present in the file are updated; missing experts use the
    model's existing alpha_default.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    t0 = time.perf_counter()
    for layer_idx, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, _HybridPagedMLP):
            continue
        key = str(layer_idx)
        if key not in data:
            continue
        layer_data = data[key]
        for expert_idx_str, alpha_val in layer_data.items():
            expert_idx = int(expert_idx_str)
            layer.mlp.alphas[expert_idx] = float(alpha_val)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms


def reset_alpha_profile(model) -> None:
    """Reset all MoE layer alphas to their trained default (alpha_default)."""
    for layer_idx, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp, _HybridPagedMLP):
            continue
        mlp = layer.mlp
        for e in range(mlp.n_experts):
            mlp.alphas[e] = mlp.alpha_default


def list_profiles(directory: str) -> List[Dict[str, Any]]:
    """Return metadata for every *.json alpha profile in directory."""
    dir_path = Path(directory)
    profiles = []
    for json_file in sorted(dir_path.glob("*_profile.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            meta = data.get("__meta__", {})
            profiles.append({
                "path": str(json_file),
                "name": json_file.stem.replace("_profile", ""),
                "n_moe_layers": meta.get("n_moe_layers"),
                "n_experts": meta.get("n_experts"),
                "label": meta.get("label", ""),
            })
        except Exception:
            pass
    return profiles


# ---------------------------------------------------------------------------
# Experiment 2: profile switching test
# ---------------------------------------------------------------------------

def run_experiment_2(
    loaded,
    output_dir: str = ".",
    *,
    max_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Experiment 2: Domain mode switching via alpha recipes.

    Expects medical_profile.json / coding_profile.json / legal_profile.json
    in output_dir (produced by Experiment 1).  Tests rapid profile switching
    and generates domain-specific text with each profile.

    Returns summary dict with generation samples and switch timing.
    """
    check_ram(15.0, "exp2-start")
    out_path = Path(output_dir)

    # Check that profiles from Experiment 1 exist
    required = ["medical_profile.json", "coding_profile.json", "legal_profile.json"]
    missing = [f for f in required if not (out_path / f).exists()]
    if missing:
        return {
            "error": f"Missing profiles from Experiment 1: {missing}",
            "success": False,
        }

    domain_configs = [
        ("medical", "Explain the treatment options for hypertension in elderly patients."),
        ("coding",  "Write a Python function to find the longest common subsequence."),
        ("legal",   "Draft an indemnification clause for a software license agreement."),
    ]

    results: Dict[str, Any] = {}
    switch_times_ms: List[float] = []

    for domain_name, prompt in domain_configs:
        profile_path = str(out_path / f"{domain_name}_profile.json")

        t_switch = load_alpha_profile(loaded.model, profile_path)
        switch_times_ms.append(t_switch)

        prompt_ids = _tokenize(loaded, prompt)
        if not prompt_ids:
            results[domain_name] = {"error": "tokenisation failed"}
            continue

        gen_ids = _generate_ids(loaded, prompt_ids, max_tokens=max_tokens)
        gen_text = loaded.tokenizer.decode(gen_ids)

        print(f"\n[Exp 2 / {domain_name}] switch={t_switch:.3f}ms")
        print(f"  Prompt:   {prompt}")
        print(f"  Response: {gen_text[:120]}")

        results[domain_name] = {
            "prompt": prompt,
            "response": gen_text,
            "switch_ms": t_switch,
        }

    reset_alpha_profile(loaded.model)

    mean_switch_ms = sum(switch_times_ms) / max(len(switch_times_ms), 1)
    max_switch_ms = max(switch_times_ms) if switch_times_ms else 0.0
    print(f"\n[Exp 2] Mean switch time: {mean_switch_ms:.3f} ms  Max: {max_switch_ms:.3f} ms")

    results["mean_switch_ms"] = mean_switch_ms
    results["max_switch_ms"] = max_switch_ms
    results["success"] = max_switch_ms < 1.0  # target: <1 ms

    # Save summary
    summary_path = out_path / "profile_switch_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")

    check_ram(15.0, "exp2-end")
    return results
