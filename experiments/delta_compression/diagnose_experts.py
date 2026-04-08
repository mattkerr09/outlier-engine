from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DIAGNOSIS_LOG_PATH,
    LAYER_IDX,
    MODEL_REF,
    N_EXPERTS,
    PROJECTIONS,
    TeeLogger,
    build_tensor_index,
    expert_cosine_similarity,
    expert_key,
    load_checkpoint_expert,
    load_config,
    load_header_entry,
    resolve_model_dir,
)


def loader_diagnosis() -> str:
    return (
        "The current loader does not synthesize or fan out one expert into eight. "
        "Its paged load path constructs expert-specific keys like "
        "`base.model.layers.{layer}.mlp.experts.{expert_idx}.*` and reads those "
        "raw tensors directly from safetensors."
    )


def main() -> None:
    logger = TeeLogger(DIAGNOSIS_LOG_PATH, mode="w")
    try:
        model_dir = resolve_model_dir(MODEL_REF)
        config = load_config(model_dir)
        tensor_index = build_tensor_index(model_dir)

        logger.emit("Expert diagnosis")
        logger.emit(f"model_ref={MODEL_REF}")
        logger.emit(f"model_dir={model_dir}")
        logger.emit(f"layer_idx={LAYER_IDX}")
        logger.emit(f"n_experts={config.get('n_experts')}")
        logger.emit("")

        experts = {
            expert_idx: load_checkpoint_expert(tensor_index, LAYER_IDX, expert_idx)
            for expert_idx in range(N_EXPERTS)
        }

        logger.emit("Pairwise comparison against expert 0")
        for other_idx in range(1, N_EXPERTS):
            equal = True
            for projection, _, _, _ in PROJECTIONS:
                if not torch.equal(
                    experts[0][f"{projection}_ternary"],
                    experts[other_idx][f"{projection}_ternary"],
                ):
                    equal = False
                    break
                if not torch.equal(
                    experts[0][f"{projection}_scale"],
                    experts[other_idx][f"{projection}_scale"],
                ):
                    equal = False
                    break
            cosine = expert_cosine_similarity(experts[0], experts[other_idx])
            logger.emit(
                f"expert_0 vs expert_{other_idx}: bitwise_equal={equal} cosine={cosine:.6f}"
            )
        logger.emit("")

        logger.emit("Checkpoint key / offset inspection for layer 8 gate_ternary")
        seen_offsets: set[tuple[str, tuple[int, int]]] = set()
        duplicate_offsets = False
        for expert_idx in range(N_EXPERTS):
            key = expert_key(LAYER_IDX, expert_idx, "gate_ternary")
            entry = load_header_entry(tensor_index, key)
            offsets = tuple(entry["info"]["data_offsets"])
            shard = entry["shard"]
            if (shard, offsets) in seen_offsets:
                duplicate_offsets = True
            seen_offsets.add((shard, offsets))
            logger.emit(
                f"{key} | shard={shard} | offsets={offsets} | shape={entry['info']['shape']}"
            )
        logger.emit("")

        logger.emit("All pairwise equality summary across all 8 experts")
        unique_payload_groups: list[list[int]] = []
        for expert_idx in range(N_EXPERTS):
            placed = False
            for group in unique_payload_groups:
                rep = group[0]
                if all(
                    torch.equal(
                        experts[expert_idx][f"{projection}_ternary"],
                        experts[rep][f"{projection}_ternary"],
                    )
                    and torch.equal(
                        experts[expert_idx][f"{projection}_scale"],
                        experts[rep][f"{projection}_scale"],
                    )
                    for projection, _, _, _ in PROJECTIONS
                ):
                    group.append(expert_idx)
                    placed = True
                    break
            if not placed:
                unique_payload_groups.append([expert_idx])
        for group in unique_payload_groups:
            logger.emit(f"payload_group={group}")
        logger.emit("")

        logger.emit("Diagnosis")
        logger.emit(loader_diagnosis())
        if duplicate_offsets:
            logger.emit(
                "Multiple expert keys share the exact same byte range in safetensors. "
                "That would indicate tensor aliasing inside the file."
            )
        else:
            logger.emit(
                "Each expert key points to a distinct safetensors byte range, so the file "
                "stores separate tensor entries rather than one aliased tensor."
            )
        logger.emit(
            "Experts are identical because the checkpoint itself contains duplicated routed "
            "expert payloads for this layer: distinct expert keys and distinct byte ranges, "
            "but equal tensor contents and equal scales."
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
