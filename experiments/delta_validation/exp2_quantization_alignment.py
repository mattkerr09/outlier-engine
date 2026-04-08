"""
Experiment 2: Independent vs joint quantization comparison.

Tests whether aligning the quantization threshold between shared and expert
weights recovers higher delta sparsity than independent quantization.

Three methods:
  A. Independent: each weight matrix quantized with its own absmean threshold
  B. Joint: shared + expert use the same global threshold (from combined stats)
  C. Per-row joint: per-row threshold from concatenated shared[i] & expert[i]

Since V1 experts are bitwise clones, we test on:
  1. The actual V1 weights (expert ternary*scale vs shared FP16)
  2. Synthetic diverse experts (shared + gaussian perturbation) to simulate
     what a properly trained V3 would look like

Uses V1 checkpoint (Outlier-Ai/Outlier-10B) — the only MoE checkpoint.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V1_REPO = "Outlier-Ai/Outlier-10B"
N_EXPERTS = 8
PROJS = ["gate", "up", "down"]
# Test on early, middle, late MoE layers
TEST_LAYERS = [2, 14, 25]
SCALE_FACTOR = 0.7  # absmean ternary threshold multiplier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_dir(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    local = Path(repo).expanduser()
    if local.exists():
        return local.resolve()
    return Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"])).resolve()


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    state = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for k in f.keys():
                state[k] = f.get_tensor(k)
    return state


def expert_float(state: dict, layer: int, expert_idx: int, proj: str) -> torch.Tensor:
    prefix = f"base.model.layers.{layer}.mlp.experts.{expert_idx}"
    ternary = state[f"{prefix}.{proj}_ternary"].float()
    scale = state[f"{prefix}.{proj}_scale"].float()
    return ternary * scale


def shared_float(state: dict, layer: int, proj: str) -> torch.Tensor:
    key = f"base.model.layers.{layer}.mlp.shared_expert.{proj}_W"
    return state[key].float()


# ---------------------------------------------------------------------------
# Quantization methods
# ---------------------------------------------------------------------------

def absmean_ternary(W: torch.Tensor, scale: float = 0.7) -> torch.Tensor:
    """Standard absmean ternary quantization."""
    threshold = scale * W.abs().mean()
    Q = torch.zeros_like(W)
    Q[W > threshold] = 1.0
    Q[W < -threshold] = -1.0
    return Q


def method_a_independent(shared: torch.Tensor, expert: torch.Tensor) -> dict:
    """Method A: Independent quantization — each matrix uses its own threshold."""
    Q_shared = absmean_ternary(shared)
    Q_expert = absmean_ternary(expert)
    delta = Q_expert - Q_shared
    sparsity = (delta == 0).float().mean().item()
    return {"method": "A_independent", "sparsity": sparsity, "delta": delta}


def method_b_joint(shared: torch.Tensor, expert: torch.Tensor) -> dict:
    """Method B: Joint global threshold from concatenated weights."""
    combined = torch.cat([shared.flatten(), expert.flatten()])
    threshold = SCALE_FACTOR * combined.abs().mean()

    Q_shared = torch.zeros_like(shared)
    Q_shared[shared > threshold] = 1.0
    Q_shared[shared < -threshold] = -1.0

    Q_expert = torch.zeros_like(expert)
    Q_expert[expert > threshold] = 1.0
    Q_expert[expert < -threshold] = -1.0

    delta = Q_expert - Q_shared
    sparsity = (delta == 0).float().mean().item()
    return {"method": "B_joint_global", "sparsity": sparsity, "delta": delta}


def method_c_per_row(shared: torch.Tensor, expert: torch.Tensor) -> dict:
    """Method C: Per-row joint threshold from concatenated row pairs."""
    assert shared.shape == expert.shape
    Q_shared = torch.zeros_like(shared)
    Q_expert = torch.zeros_like(expert)

    for i in range(shared.shape[0]):
        row_combined = torch.cat([shared[i], expert[i]])
        threshold = SCALE_FACTOR * row_combined.abs().mean()

        Q_shared[i][shared[i] > threshold] = 1.0
        Q_shared[i][shared[i] < -threshold] = -1.0

        Q_expert[i][expert[i] > threshold] = 1.0
        Q_expert[i][expert[i] < -threshold] = -1.0

    delta = Q_expert - Q_shared
    sparsity = (delta == 0).float().mean().item()
    return {"method": "C_per_row_joint", "sparsity": sparsity, "delta": delta}


# ---------------------------------------------------------------------------
# Synthetic diverse expert generator
# ---------------------------------------------------------------------------

def make_synthetic_expert(shared: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    """Create a synthetic diverse expert = shared + gaussian noise."""
    return shared + torch.randn_like(shared) * noise_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EXPERIMENT 2: Independent vs Joint Quantization Comparison")
    print("=" * 70)

    v1_dir = resolve_model_dir(V1_REPO)
    print("Loading V1 safetensors...")
    state = load_safetensors(v1_dir)
    print(f"Loaded {len(state)} keys\n")

    # -----------------------------------------------------------------------
    # Part 1: Test on ACTUAL V1 weights (expert_float vs shared_float)
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PART 1: Actual V1 weights (expert ternary*scale vs shared FP16)")
    print("=" * 70)
    print(f"{'Layer':>5} {'Proj':>5} {'Expert':>6} | {'A_indep':>10} {'B_joint':>10} {'C_perrow':>10} | {'B-A':>7} {'C-A':>7}")
    print("-" * 80)

    part1_results = []
    for layer_idx in TEST_LAYERS:
        for proj in PROJS:
            s = shared_float(state, layer_idx, proj)
            for j in range(N_EXPERTS):
                e = expert_float(state, layer_idx, j, proj)
                ra = method_a_independent(s, e)
                rb = method_b_joint(s, e)
                rc = method_c_per_row(s, e)

                diff_ba = rb["sparsity"] - ra["sparsity"]
                diff_ca = rc["sparsity"] - ra["sparsity"]

                if j == 0:  # Print only first expert (all clones)
                    print(f"{layer_idx:>5} {proj:>5} {j:>6} | "
                          f"{ra['sparsity']:>9.2%} {rb['sparsity']:>9.2%} {rc['sparsity']:>9.2%} | "
                          f"{diff_ba:>+6.2%} {diff_ca:>+6.2%}")

                part1_results.append({
                    "layer": layer_idx, "proj": proj, "expert": j,
                    "A": ra["sparsity"], "B": rb["sparsity"], "C": rc["sparsity"],
                })

    avg_a = sum(r["A"] for r in part1_results) / len(part1_results)
    avg_b = sum(r["B"] for r in part1_results) / len(part1_results)
    avg_c = sum(r["C"] for r in part1_results) / len(part1_results)
    print(f"\nPart 1 Averages (all experts, 3 layers):")
    print(f"  Method A (independent): {avg_a:.4%}")
    print(f"  Method B (joint global): {avg_b:.4%}")
    print(f"  Method C (per-row joint): {avg_c:.4%}")
    print(f"  B improvement over A: {avg_b - avg_a:+.4%}")
    print(f"  C improvement over A: {avg_c - avg_a:+.4%}")

    # -----------------------------------------------------------------------
    # Part 2: Test on SYNTHETIC diverse experts
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: Synthetic diverse experts (shared + noise, std=0.02)")
    print("=" * 70)
    print("This simulates what properly-trained V3 experts might look like.\n")

    noise_levels = [0.005, 0.01, 0.02, 0.05]
    print(f"{'Noise':>7} | {'A_indep':>10} {'B_joint':>10} {'C_perrow':>10} | {'B-A':>7} {'C-A':>7}")
    print("-" * 70)

    for noise_std in noise_levels:
        torch.manual_seed(42)
        synth_results = []
        for layer_idx in TEST_LAYERS:
            for proj in PROJS:
                s = shared_float(state, layer_idx, proj)
                for j in range(4):  # 4 synthetic experts per config
                    e = make_synthetic_expert(s, noise_std=noise_std)
                    ra = method_a_independent(s, e)
                    rb = method_b_joint(s, e)
                    rc = method_c_per_row(s, e)
                    synth_results.append({
                        "A": ra["sparsity"], "B": rb["sparsity"], "C": rc["sparsity"],
                    })

        sa = sum(r["A"] for r in synth_results) / len(synth_results)
        sb = sum(r["B"] for r in synth_results) / len(synth_results)
        sc = sum(r["C"] for r in synth_results) / len(synth_results)
        print(f"{noise_std:>7.3f} | {sa:>9.2%} {sb:>9.2%} {sc:>9.2%} | {sb - sa:>+6.2%} {sc - sa:>+6.2%}")

    # -----------------------------------------------------------------------
    # Part 3: Delta value distribution analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: Delta value distribution (ternary delta can be {-2,-1,0,1,2})")
    print("=" * 70)

    for layer_idx in [TEST_LAYERS[1]]:  # middle layer only
        for proj in PROJS:
            s = shared_float(state, layer_idx, proj)
            e = expert_float(state, layer_idx, 0, proj)

            for method_name, method_fn in [("A", method_a_independent), ("B", method_b_joint)]:
                result = method_fn(s, e)
                delta = result["delta"]
                counts = {}
                for val in [-2, -1, 0, 1, 2]:
                    counts[val] = (delta == val).sum().item()
                total = delta.numel()
                print(f"  Layer {layer_idx} {proj:>4} Method {method_name}: ", end="")
                for val in [-2, -1, 0, 1, 2]:
                    pct = counts[val] / total * 100
                    print(f"[{val:+d}]={pct:5.1f}% ", end="")
                print()
        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("KEY ANSWERS")
    print("=" * 70)
    print(f"\nQ: Does joint quantization recover meaningful sparsity?")
    print(f"A: On actual V1 weights (expert = ternary*scale, shared = FP16):")
    print(f"   Independent (A): {avg_a:.2%} sparsity")
    print(f"   Joint global (B): {avg_b:.2%} sparsity")
    print(f"   Per-row joint (C): {avg_c:.2%} sparsity")
    improvement = max(avg_b - avg_a, avg_c - avg_a)
    if improvement > 0.05:
        print(f"   YES — joint quantization improves sparsity by up to {improvement:+.2%}")
    elif improvement > 0.01:
        print(f"   MARGINAL — improvement is only {improvement:+.2%}")
    else:
        print(f"   NO — improvement is negligible ({improvement:+.2%})")
    print(f"\n   NOTE: V1 experts are clones. This result reflects quantization")
    print(f"   alignment only. Real gains require diverse trained experts (V3).")
    print(f"   Synthetic tests with noise_std=0.02 show the method's potential")
    print(f"   for properly trained experts.")


if __name__ == "__main__":
    main()
