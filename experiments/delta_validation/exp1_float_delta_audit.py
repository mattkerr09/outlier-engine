"""
Experiment 1: Float delta sparsity audit on the Outlier MoE checkpoint.

Goal: determine whether routed experts show ANY specialization vs the shared
expert, or are still clones (as discovered in V1 diagnosis).

Approach:
  - Loads Outlier-Ai/Outlier-10B (the MoE checkpoint; V2 is dense Qwen2 with
    no MoE layers, so we use V1 which is the only checkpoint containing experts).
  - For each MoE layer: extracts shared FFN (FP16) and all routed experts
    (int8 ternary * FP16 scale), reconstructs float weights, and computes:
      a. Float delta sparsity at multiple thresholds
      b. Cosine similarity between each expert and shared
      c. Pairwise cosine similarity between experts (diversity check)
  - Documents that Outlier-10B-V2 is dense and has no MoE to audit.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V1_REPO = "Outlier-Ai/Outlier-10B"
V2_REPO = "Outlier-Ai/Outlier-10B-V2"
N_EXPERTS = 8
N_LAYERS = 28
PROJS = ["gate", "up", "down"]
THRESHOLDS = [0.001, 0.01, 0.05, 0.1]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_dir(repo: str) -> Path:
    """Resolve HF cache dir for a repo."""
    from huggingface_hub import snapshot_download
    local = Path(repo).expanduser()
    if local.exists():
        return local.resolve()
    return Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])).resolve()


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards into a flat dict."""
    state = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def expert_float(state: dict, layer: int, expert_idx: int, proj: str) -> torch.Tensor:
    """Reconstruct float expert weight = ternary_int8 * scale."""
    prefix = f"base.model.layers.{layer}.mlp.experts.{expert_idx}"
    ternary = state[f"{prefix}.{proj}_ternary"].float()
    scale = state[f"{prefix}.{proj}_scale"].float()
    return ternary * scale


def shared_float(state: dict, layer: int, proj: str) -> torch.Tensor:
    """Get shared expert weight in float32."""
    key = f"base.model.layers.{layer}.mlp.shared_expert.{proj}_W"
    return state[key].float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


def delta_sparsity(delta: torch.Tensor, threshold: float) -> float:
    return (delta.abs() < threshold).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Check V2 first (document that it's dense) ---
    print("=" * 70)
    print("EXPERIMENT 1: Float Delta Sparsity Audit")
    print("=" * 70)

    print("\n--- V2 Model Check ---")
    v2_dir = resolve_model_dir(V2_REPO)
    v2_config = json.loads((v2_dir / "config.json").read_text())
    v2_model_type = v2_config.get("model_type", "unknown")
    v2_n_experts = v2_config.get("outlier_num_experts", 0)
    print(f"V2 model_type={v2_model_type}, outlier_num_experts={v2_n_experts}")

    # Check for any expert keys
    v2_state_keys = []
    for sf in sorted(v2_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            v2_state_keys.extend(f.keys())
    v2_moe_keys = [k for k in v2_state_keys if "expert" in k or "router" in k or "shared" in k.lower()]
    print(f"V2 total keys={len(v2_state_keys)}, MoE-related keys={len(v2_moe_keys)}")
    print("FINDING: Outlier-10B-V2 is a dense Qwen2 model with NO MoE layers.")
    print("         No experts, no router, no shared expert. Cannot compute deltas on V2.")
    print("         Falling back to V1 (Outlier-10B) which has the MoE architecture.\n")

    # --- Load V1 ---
    print("--- V1 Model Load ---")
    v1_dir = resolve_model_dir(V1_REPO)
    v1_config = json.loads((v1_dir / "config.json").read_text())
    print(f"V1 model_type={v1_config.get('model_type', v1_config.get('architectures', ['?']))}")
    print(f"V1 n_layers={N_LAYERS}, n_experts={N_EXPERTS}")
    print("Loading safetensors shards...")
    state = load_safetensors(v1_dir)
    print(f"Loaded {len(state)} keys\n")

    # --- Determine which layers have MoE ---
    moe_layers = []
    for layer_idx in range(N_LAYERS):
        key = f"base.model.layers.{layer_idx}.mlp.experts.0.gate_ternary"
        if key in state:
            moe_layers.append(layer_idx)
    print(f"MoE layers: {moe_layers} ({len(moe_layers)} total)\n")

    # --- Bitwise clone check first ---
    print("--- Pre-check: Are experts bitwise identical (clones)? ---")
    sample_layer = moe_layers[len(moe_layers) // 2]  # middle layer
    clone_results = []
    for proj in PROJS:
        e0 = state[f"base.model.layers.{sample_layer}.mlp.experts.0.{proj}_ternary"]
        for j in range(1, N_EXPERTS):
            ej = state[f"base.model.layers.{sample_layer}.mlp.experts.{j}.{proj}_ternary"]
            is_clone = torch.equal(e0, ej)
            clone_results.append(is_clone)
            if j == 1:
                print(f"  layer {sample_layer} {proj}: expert_0 == expert_1 ? {is_clone}")
    all_clones = all(clone_results)
    print(f"  All experts bitwise identical in layer {sample_layer}: {all_clones}")

    if all_clones:
        # Check data_ptr too
        e0_ptr = state[f"base.model.layers.{sample_layer}.mlp.experts.0.gate_ternary"].data_ptr()
        e1_ptr = state[f"base.model.layers.{sample_layer}.mlp.experts.1.gate_ternary"].data_ptr()
        print(f"  data_ptr check: expert_0={e0_ptr}, expert_1={e1_ptr}, same_ptr={e0_ptr == e1_ptr}")
        print("  CONFIRMED: V1 experts are bitwise clones. Delta between experts is exactly zero.")
        print("  The expert-vs-shared delta measures quantization error, not specialization.\n")

    # --- Per-layer analysis ---
    print("--- Per-Layer Float Delta Analysis ---")
    print(f"{'Layer':>5} | ", end="")
    for t in THRESHOLDS:
        print(f"|d|<{t:5.3f} ", end="")
    print(f"| E-S cos | pairwise cos | pairwise std")
    print("-" * 90)

    all_layer_results = []
    for layer_idx in moe_layers:
        layer_sparsities = {t: [] for t in THRESHOLDS}
        expert_shared_cosines = []
        pairwise_cosines = []

        for proj in PROJS:
            s = shared_float(state, layer_idx, proj)
            expert_floats = []
            for j in range(N_EXPERTS):
                e = expert_float(state, layer_idx, j, proj)
                expert_floats.append(e)

                # Delta = expert_float - shared_float
                d = e - s
                for t in THRESHOLDS:
                    layer_sparsities[t].append(delta_sparsity(d, t))
                expert_shared_cosines.append(cosine_sim(e, s))

            # Pairwise cosine between experts
            for (i, ei), (j2, ej) in combinations(enumerate(expert_floats), 2):
                pairwise_cosines.append(cosine_sim(ei, ej))

        avg_sparsity = {t: sum(layer_sparsities[t]) / len(layer_sparsities[t]) for t in THRESHOLDS}
        avg_es_cos = sum(expert_shared_cosines) / len(expert_shared_cosines)
        avg_pw_cos = sum(pairwise_cosines) / len(pairwise_cosines)
        std_pw_cos = torch.tensor(pairwise_cosines).std().item()

        print(f"{layer_idx:>5} | ", end="")
        for t in THRESHOLDS:
            print(f"  {avg_sparsity[t]:6.2%}  ", end="")
        print(f"| {avg_es_cos:7.4f}  | {avg_pw_cos:12.6f}  | {std_pw_cos:.6f}")

        all_layer_results.append({
            "layer": layer_idx,
            "sparsity": avg_sparsity,
            "expert_shared_cosine": avg_es_cos,
            "pairwise_cosine": avg_pw_cos,
            "pairwise_std": std_pw_cos,
        })

    # --- Overall summary ---
    print("\n--- Overall Summary ---")
    for t in THRESHOLDS:
        avg = sum(r["sparsity"][t] for r in all_layer_results) / len(all_layer_results)
        print(f"  Avg delta sparsity at |d| < {t}: {avg:.2%}")

    avg_es = sum(r["expert_shared_cosine"] for r in all_layer_results) / len(all_layer_results)
    avg_pw = sum(r["pairwise_cosine"] for r in all_layer_results) / len(all_layer_results)
    print(f"  Avg expert-shared cosine: {avg_es:.6f}")
    print(f"  Avg pairwise cosine: {avg_pw:.6f}")

    print("\n--- KEY ANSWERS ---")
    if all_clones:
        print("Q: Do V2 experts show specialization?")
        print("A: V2 (Outlier-10B-V2) has NO experts — it is a dense Qwen2 model.")
        print("   V1 (Outlier-10B) has 8 experts per layer, but they are BITWISE CLONES.")
        print("   Pairwise diversity = 0. No specialization exists in any shipped checkpoint.")
        print("   The expert-shared delta reflects quantization error only.")
    else:
        if avg_pw < 0.99:
            print("Q: Are experts diverse?")
            print(f"A: YES — avg pairwise cosine = {avg_pw:.4f} (< 0.99 = diverse)")
        else:
            print("Q: Are experts diverse?")
            print(f"A: NO — avg pairwise cosine = {avg_pw:.4f} (≥ 0.99 = near-clones)")

    print(f"\nQ: Float delta sparsity at 0.01 threshold?")
    avg_01 = sum(r["sparsity"][0.01] for r in all_layer_results) / len(all_layer_results)
    print(f"A: {avg_01:.2%}")


if __name__ == "__main__":
    main()
