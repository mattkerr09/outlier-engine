from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DIVERSE_EXPERTS_DIR,
    EXPERIMENT_V2_LOG_PATH,
    LAYER_IDX,
    MODEL_REF,
    N_EXPERTS,
    PROJECTIONS,
    RESULTS_PATH,
    TeeLogger,
    bytes_to_mib,
    diverse_expert_path,
    expert_cosine_similarity,
    float16_nbytes,
    full_expert_tq10_nbytes,
    load_safetensors_file,
    nonzero_stream_tq10_nbytes,
    pack_bits_to_bytes,
    position_index_bits,
)


TRANSITIONS = {
    (-1, 0): "-1->0",
    (-1, 1): "-1->+1",
    (0, -1): "0->-1",
    (0, 1): "0->+1",
    (1, 0): "+1->0",
    (1, -1): "+1->-1",
}


@dataclass
class ExpertV2Result:
    expert_idx: int
    delta_sparsity: float
    reconstruction_error: float
    cosine_similarity: float
    full_tq10_bytes: int
    dense_delta_tq10_bytes: int
    sparse_ternary_delta_bytes: int
    sparse_raw_delta_bytes: int
    diff_fraction: float
    sparse_transition_bytes: int
    transition_counts: dict[str, int]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_expert(
    shared_tensors: dict[str, torch.Tensor],
    expert_tensors: dict[str, torch.Tensor],
    expert_idx: int,
) -> ExpertV2Result:
    zero_count = 0
    total_elements = 0
    error_sq_sum = 0.0
    orig_sq_sum = 0.0
    recon_sq_sum = 0.0
    dot_sum = 0.0

    full_tq10_bytes = full_expert_tq10_nbytes(expert_tensors)
    dense_delta_tq10_bytes = 0
    sparse_ternary_delta_bytes = 0
    sparse_raw_delta_bytes = 0
    sparse_transition_bytes = 0
    transition_counts = {label: 0 for label in TRANSITIONS.values()}

    for projection, _, _, _ in PROJECTIONS:
        shared_q = shared_tensors[f"{projection}_ternary"].to(torch.int16)
        expert_q = expert_tensors[f"{projection}_ternary"].to(torch.int16)
        expert_scale = expert_tensors[f"{projection}_scale"]

        numel = expert_q.numel()
        total_elements += numel

        delta = expert_q - shared_q
        zero_count += int((delta == 0).sum().item())

        delta_scale = delta.float().abs().mean().clamp(min=1e-8)
        dense_delta_tq10_bytes += math.ceil(numel / 5) + float16_nbytes(1)

        delta_q = (delta.float() / delta_scale).round().clamp(-1, 1).to(torch.int8)
        recon = shared_q.float() + delta_q.float() * float(delta_scale.item())
        orig = expert_q.float()

        error_sq_sum += (orig - recon).square().sum().item()
        orig_sq_sum += orig.square().sum().item()
        recon_sq_sum += recon.square().sum().item()
        dot_sum += (orig * recon).sum().item()

        diff_mask = delta != 0
        diff_count = int(diff_mask.sum().item())
        pos_bits = position_index_bits(numel)

        sparse_ternary_delta_bytes += (
            pack_bits_to_bytes(diff_count * pos_bits)
            + nonzero_stream_tq10_nbytes(delta_q[diff_mask])
            + float16_nbytes(1)
        )
        sparse_raw_delta_bytes += (
            pack_bits_to_bytes(diff_count * pos_bits)
            + pack_bits_to_bytes(diff_count * 2)
        )
        sparse_transition_bytes += (
            pack_bits_to_bytes(diff_count * (pos_bits + 3))
            + expert_scale.numel() * expert_scale.element_size()
        )

        shared_flat = shared_q.reshape(-1)
        expert_flat = expert_q.reshape(-1)
        for (src, dst), label in TRANSITIONS.items():
            transition_counts[label] += int(
                ((shared_flat == src) & (expert_flat == dst)).sum().item()
            )

    reconstruction_error = math.sqrt(error_sq_sum)
    cosine_similarity = dot_sum / max(
        math.sqrt(orig_sq_sum) * math.sqrt(recon_sq_sum),
        1e-12,
    )
    delta_sparsity = zero_count / total_elements
    diff_fraction = 1.0 - delta_sparsity

    return ExpertV2Result(
        expert_idx=expert_idx,
        delta_sparsity=delta_sparsity,
        reconstruction_error=reconstruction_error,
        cosine_similarity=cosine_similarity,
        full_tq10_bytes=full_tq10_bytes,
        dense_delta_tq10_bytes=dense_delta_tq10_bytes,
        sparse_ternary_delta_bytes=sparse_ternary_delta_bytes,
        sparse_raw_delta_bytes=sparse_raw_delta_bytes,
        diff_fraction=diff_fraction,
        sparse_transition_bytes=sparse_transition_bytes,
        transition_counts=transition_counts,
    )


def update_results_markdown(
    pairwise_cosines: list[float],
    results: list[ExpertV2Result],
) -> None:
    avg_pairwise = average(pairwise_cosines)
    min_pairwise = min(pairwise_cosines)
    max_pairwise = max(pairwise_cosines)

    avg_delta_sparsity = average([item.delta_sparsity for item in results])
    avg_recon = average([item.reconstruction_error for item in results])
    avg_cos = average([item.cosine_similarity for item in results])
    avg_dense_ratio = average(
        [item.full_tq10_bytes / item.dense_delta_tq10_bytes for item in results]
    )
    avg_sparse_ternary_ratio = average(
        [item.full_tq10_bytes / item.sparse_ternary_delta_bytes for item in results]
    )
    avg_sparse_raw_ratio = average(
        [item.full_tq10_bytes / item.sparse_raw_delta_bytes for item in results]
    )
    avg_transition_ratio = average(
        [item.full_tq10_bytes / item.sparse_transition_bytes for item in results]
    )
    avg_diff_fraction = average([item.diff_fraction for item in results])

    best_avg_ratio = max(
        avg_dense_ratio,
        avg_sparse_ternary_ratio,
        avg_sparse_raw_ratio,
        avg_transition_ratio,
    )
    if best_avg_ratio <= 1.0:
        claim_statement = (
            "No. Under this diverse-expert synthetic test, none of the measured delta "
            "encodings compressed better than the full TQ1_0 expert."
        )
    elif best_avg_ratio < 2.0:
        claim_statement = (
            f"No. The best measured average ratio was only `{best_avg_ratio:.3f}x`, "
            "far below an 8-16x claim."
        )
    else:
        claim_statement = (
            f"No for 8-16x. The best measured average ratio was `{best_avg_ratio:.3f}x`, "
            "which is interesting but still well short of the claimed range."
        )

    recommendation = (
        "This does not look worth pursuing as a broad patent non-provisional compression "
        "claim in its current form. The real bottleneck is position encoding: once experts "
        "are truly diverse, the sparse diff is too dense for index overhead to amortize."
    )
    if best_avg_ratio > 1.0:
        recommendation = (
            "There may still be a narrower idea around structured or block-sparse diffs, "
            "but this exact per-weight sparse-delta framing does not support an 8-16x claim."
        )

    lines = [
        "# Delta Compression Results",
        "",
        "## V1 Diagnosis",
        "",
        f"Checkpoint diagnosis was run on `{MODEL_REF}` layer `{LAYER_IDX}`.",
        "The routed experts are duplicated in the checkpoint itself, not duplicated by the loader.",
        "The safetensors file contains distinct expert keys and distinct byte ranges, but the payloads are bitwise identical.",
        "",
        "## V2 Diverse Experts",
        "",
        (
            f"Created `{N_EXPERTS}` synthetic routed experts for layer `{LAYER_IDX}` by applying "
            "dropout upcycling (`r=0.5`) to the shared FFN and quantizing each expert with scalar absmean ternarization."
        ),
        f"- Average pairwise cosine similarity across synthesized experts: `{avg_pairwise:.6f}`",
        f"- Min / max pairwise cosine similarity: `{min_pairwise:.6f}` / `{max_pairwise:.6f}`",
        "",
        "## V2 Compression Findings",
        "",
        "Reconstruction metrics below are measured in ternary-code space because the delta is defined on ternary shared/expert weights.",
        f"- Average delta sparsity (`expert_ternary - shared_ternary == 0`): `{avg_delta_sparsity * 100:.2f}%`",
        f"- Average differing positions vs shared ternary: `{avg_diff_fraction * 100:.2f}%`",
        f"- Average reconstruction L2: `{avg_recon:.4f}`",
        f"- Average cosine similarity: `{avg_cos:.6f}`",
        f"- Average dense delta TQ1_0 ratio: `{avg_dense_ratio:.3f}x`",
        f"- Average sparse ternary delta ratio (indices + packed delta values): `{avg_sparse_ternary_ratio:.3f}x`",
        f"- Average sparse raw delta ratio (indices + 2-bit delta values): `{avg_sparse_raw_ratio:.3f}x`",
        f"- Average sparse transition-diff ratio (indices + 3-bit transition code): `{avg_transition_ratio:.3f}x`",
        "",
        "## Claim 10b",
        "",
        claim_statement,
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Per-Expert V2 Results",
        "",
        "| Expert | Delta Sparsity | Recon L2 | Cosine | Full TQ1_0 MiB | Dense Delta Ratio | Sparse Ternary Ratio | Sparse Raw Ratio | Diff % | Transition Ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in results:
        lines.append(
            "| "
            f"{item.expert_idx} | "
            f"{item.delta_sparsity * 100:.2f}% | "
            f"{item.reconstruction_error:.4f} | "
            f"{item.cosine_similarity:.6f} | "
            f"{bytes_to_mib(item.full_tq10_bytes):.2f} | "
            f"{item.full_tq10_bytes / item.dense_delta_tq10_bytes:.3f}x | "
            f"{item.full_tq10_bytes / item.sparse_ternary_delta_bytes:.3f}x | "
            f"{item.full_tq10_bytes / item.sparse_raw_delta_bytes:.3f}x | "
            f"{item.diff_fraction * 100:.2f}% | "
            f"{item.full_tq10_bytes / item.sparse_transition_bytes:.3f}x |"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logger = TeeLogger(EXPERIMENT_V2_LOG_PATH, mode="a")
    try:
        logger.emit("")
        logger.emit("Delta experiment v2")
        logger.emit(f"diverse_experts_dir={DIVERSE_EXPERTS_DIR}")

        shared_tensors = load_safetensors_file(DIVERSE_EXPERTS_DIR / "shared_ternary.safetensors")
        experts = {
            expert_idx: load_safetensors_file(diverse_expert_path(expert_idx))
            for expert_idx in range(N_EXPERTS)
        }

        pairwise_cosines = []
        logger.emit("Recomputed pairwise cosine similarities")
        for expert_i in range(N_EXPERTS):
            for expert_j in range(expert_i + 1, N_EXPERTS):
                cosine = expert_cosine_similarity(experts[expert_i], experts[expert_j])
                pairwise_cosines.append(cosine)
                logger.emit(f"expert_{expert_i} vs expert_{expert_j}: cosine={cosine:.6f}")
        logger.emit("")

        results: list[ExpertV2Result] = []
        aggregate_transitions = {label: 0 for label in TRANSITIONS.values()}
        logger.emit("Per-expert delta compression results")
        for expert_idx in range(N_EXPERTS):
            result = evaluate_expert(shared_tensors, experts[expert_idx], expert_idx)
            results.append(result)
            for label, count in result.transition_counts.items():
                aggregate_transitions[label] += count
            logger.emit(
                f"expert_{expert_idx}: "
                f"delta_sparsity={result.delta_sparsity * 100:.2f}% | "
                f"recon_l2={result.reconstruction_error:.4f} | "
                f"cos={result.cosine_similarity:.6f} | "
                f"full_tq10={bytes_to_mib(result.full_tq10_bytes):.2f} MiB | "
                f"dense_ratio={result.full_tq10_bytes / result.dense_delta_tq10_bytes:.3f}x | "
                f"sparse_ternary_ratio={result.full_tq10_bytes / result.sparse_ternary_delta_bytes:.3f}x | "
                f"sparse_raw_ratio={result.full_tq10_bytes / result.sparse_raw_delta_bytes:.3f}x | "
                f"diff={result.diff_fraction * 100:.2f}% | "
                f"transition_ratio={result.full_tq10_bytes / result.sparse_transition_bytes:.3f}x"
            )
        logger.emit("")

        avg_delta_sparsity = average([item.delta_sparsity for item in results])
        avg_recon = average([item.reconstruction_error for item in results])
        avg_cos = average([item.cosine_similarity for item in results])
        avg_dense_ratio = average(
            [item.full_tq10_bytes / item.dense_delta_tq10_bytes for item in results]
        )
        avg_sparse_ternary_ratio = average(
            [item.full_tq10_bytes / item.sparse_ternary_delta_bytes for item in results]
        )
        avg_sparse_raw_ratio = average(
            [item.full_tq10_bytes / item.sparse_raw_delta_bytes for item in results]
        )
        avg_transition_ratio = average(
            [item.full_tq10_bytes / item.sparse_transition_bytes for item in results]
        )
        avg_diff_fraction = average([item.diff_fraction for item in results])

        logger.emit("Averages")
        logger.emit(
            f"average_pairwise_cosine={average(pairwise_cosines):.6f}"
        )
        logger.emit(f"average_delta_sparsity={avg_delta_sparsity * 100:.2f}%")
        logger.emit(f"average_diff_fraction={avg_diff_fraction * 100:.2f}%")
        logger.emit(f"average_reconstruction_error={avg_recon:.6f}")
        logger.emit(f"average_cosine_similarity={avg_cos:.6f}")
        logger.emit(f"average_dense_delta_ratio={avg_dense_ratio:.6f}x")
        logger.emit(f"average_sparse_ternary_delta_ratio={avg_sparse_ternary_ratio:.6f}x")
        logger.emit(f"average_sparse_raw_delta_ratio={avg_sparse_raw_ratio:.6f}x")
        logger.emit(f"average_sparse_transition_ratio={avg_transition_ratio:.6f}x")
        logger.emit("aggregate_transition_counts=" + json.dumps(aggregate_transitions, sort_keys=True))

        update_results_markdown(pairwise_cosines, results)
        logger.emit(f"results_path={RESULTS_PATH}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
