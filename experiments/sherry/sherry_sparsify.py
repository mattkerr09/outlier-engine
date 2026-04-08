from __future__ import annotations

import json
import math
from pathlib import Path

from common import (
    LAYER_IDX,
    MODEL_REF,
    N_EXPERTS,
    OUTPUT_DIR,
    PROJECTIONS,
    SPARSIFY_SUMMARY_PATH,
    build_tensor_index,
    enforce_sherry_projection,
    iter_diverse_expert_projections,
    load_config,
    load_shared_fp16,
    merge_histograms,
    resolve_model_dir,
    write_json,
)


def evaluate_expert(shared_fp16: dict, expert_idx: int) -> tuple[dict, dict[str, int]]:
    total_numel = 0
    total_groups = 0
    natural_zeros = 0
    enforced_zeros = 0
    groups_needing_enforcement = 0
    weights_forced_zero = 0
    dot_sum = 0.0
    orig_norm_sq = 0.0
    enforced_norm_sq = 0.0
    error_sq_sum = 0.0
    zero_hist_before: dict[str, int] = {}
    zero_hist_after: dict[str, int] = {}
    exact_one_zero_groups_after = 0
    multi_zero_groups_after = 0
    expert_shapes: dict[str, int] = {}
    per_projection: dict[str, dict] = {}

    for projection, original_float, ternary, scale in iter_diverse_expert_projections(shared_fp16, expert_idx):
        enforced, stats = enforce_sherry_projection(ternary, original_float)
        scale_value = float(scale.item())
        scale_sq = scale_value * scale_value

        nonzero_before = int((ternary != 0).sum().item())
        nonzero_after = int((enforced != 0).sum().item())
        forced = int(stats["weights_forced_zero"])

        total_numel += int(stats["numel"])
        total_groups += int(stats["groups"])
        natural_zeros += int(stats["natural_zeros"])
        enforced_zeros += int(stats["enforced_zeros"])
        groups_needing_enforcement += int(stats["groups_needing_enforcement"])
        weights_forced_zero += forced
        exact_one_zero_groups_after += int(stats["exact_one_zero_groups_after"])
        multi_zero_groups_after += int(stats["multi_zero_groups_after"])
        zero_hist_before = merge_histograms(zero_hist_before, stats["zero_hist_before"])
        zero_hist_after = merge_histograms(zero_hist_after, stats["zero_hist_after"])

        dot_sum += nonzero_after * scale_sq
        orig_norm_sq += nonzero_before * scale_sq
        enforced_norm_sq += nonzero_after * scale_sq
        error_sq_sum += forced * scale_sq

        expert_shapes[f"{projection}_ternary_numel"] = int(ternary.numel())
        expert_shapes[f"{projection}_scale_numel"] = int(scale.numel())
        per_projection[projection] = {
            "numel": int(stats["numel"]),
            "groups": int(stats["groups"]),
            "natural_sparsity": float(stats["natural_zeros"]) / max(int(stats["numel"]), 1),
            "enforced_sparsity": float(stats["enforced_zeros"]) / max(int(stats["numel"]), 1),
            "groups_needing_enforcement_pct": float(stats["groups_needing_enforcement"]) / max(int(stats["groups"]), 1),
            "weights_forced_zero_pct": float(forced) / max(int(stats["numel"]), 1),
            "exact_one_zero_groups_after_pct": float(stats["exact_one_zero_groups_after"]) / max(int(stats["groups"]), 1),
            "multi_zero_groups_after_pct": float(stats["multi_zero_groups_after"]) / max(int(stats["groups"]), 1),
            "scale": scale_value,
        }

    cosine = dot_sum / max(math.sqrt(orig_norm_sq) * math.sqrt(enforced_norm_sq), 1e-12)
    result = {
        "expert_idx": expert_idx,
        "numel": total_numel,
        "groups": total_groups,
        "natural_sparsity": natural_zeros / max(total_numel, 1),
        "enforced_sparsity": enforced_zeros / max(total_numel, 1),
        "groups_needing_enforcement_pct": groups_needing_enforcement / max(total_groups, 1),
        "weights_forced_zero_pct": weights_forced_zero / max(total_numel, 1),
        "exact_one_zero_groups_after_pct": exact_one_zero_groups_after / max(total_groups, 1),
        "multi_zero_groups_after_pct": multi_zero_groups_after / max(total_groups, 1),
        "cosine_similarity": cosine,
        "l2_error": math.sqrt(error_sq_sum),
        "zero_hist_before": zero_hist_before,
        "zero_hist_after": zero_hist_after,
        "projection_metrics": per_projection,
    }
    return result, expert_shapes


def average(results: list[dict], key: str) -> float:
    return sum(item[key] for item in results) / len(results)


def main() -> None:
    model_dir = resolve_model_dir(MODEL_REF)
    config = load_config(model_dir)
    tensor_index = build_tensor_index(model_dir)
    shared_fp16 = load_shared_fp16(tensor_index, layer_idx=LAYER_IDX)

    per_expert: list[dict] = []
    expert_shapes: dict[str, int] | None = None

    for expert_idx in range(N_EXPERTS):
        result, shapes = evaluate_expert(shared_fp16, expert_idx)
        per_expert.append(result)
        if expert_shapes is None:
            expert_shapes = shapes
        print(
            f"expert_{expert_idx}: natural={result['natural_sparsity'] * 100:.2f}% | "
            f"enforced={result['enforced_sparsity'] * 100:.2f}% | "
            f"groups={result['groups_needing_enforcement_pct'] * 100:.2f}% | "
            f"forced={result['weights_forced_zero_pct'] * 100:.2f}% | "
            f"cos={result['cosine_similarity']:.6f} | "
            f"l2={result['l2_error']:.4f}"
        )

    assert expert_shapes is not None
    sample = per_expert[0]
    summary = {
        "model_ref": MODEL_REF,
        "layer_idx": LAYER_IDX,
        "n_experts": N_EXPERTS,
        "assumption": (
            "Metrics use the deterministic synthetic layer-8 experts from experiments/delta_compression "
            "because the checkpointed routed experts are duplicated ternary tensors and do not store "
            "per-weight pre-rounding magnitudes."
        ),
        "config": {
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "n_experts": config.get("n_experts"),
        },
        "sample_expert": sample,
        "layer_8_average": {
            "natural_sparsity": average(per_expert, "natural_sparsity"),
            "enforced_sparsity": average(per_expert, "enforced_sparsity"),
            "groups_needing_enforcement_pct": average(per_expert, "groups_needing_enforcement_pct"),
            "weights_forced_zero_pct": average(per_expert, "weights_forced_zero_pct"),
            "exact_one_zero_groups_after_pct": average(per_expert, "exact_one_zero_groups_after_pct"),
            "multi_zero_groups_after_pct": average(per_expert, "multi_zero_groups_after_pct"),
            "cosine_similarity": average(per_expert, "cosine_similarity"),
            "l2_error": average(per_expert, "l2_error"),
        },
        "expert_shapes": expert_shapes,
        "per_expert": per_expert,
    }

    write_json(SPARSIFY_SUMMARY_PATH, summary)
    print(f"wrote {SPARSIFY_SUMMARY_PATH}")
    print(json.dumps(summary["layer_8_average"], indent=2))


if __name__ == "__main__":
    main()
