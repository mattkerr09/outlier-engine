from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent))

from common import (
    DIVERSE_EXPERTS_DIR,
    DROP_RATE,
    EXPERIMENT_V2_LOG_PATH,
    LAYER_IDX,
    MODEL_REF,
    N_EXPERTS,
    PROJECTIONS,
    TeeLogger,
    build_tensor_index,
    diverse_expert_path,
    expert_cosine_similarity,
    load_config,
    load_shared_fp16,
    quantize_absmean,
    resolve_model_dir,
    save_safetensors_file,
    shared_quantized_path,
)


def create_diverse_expert_tensors(
    shared_fp16: dict[str, torch.Tensor],
    expert_idx: int,
) -> dict[str, torch.Tensor]:
    seed = LAYER_IDX * 1000 + expert_idx
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    tensors: dict[str, torch.Tensor] = {}
    keep_prob = 1.0 - DROP_RATE
    scale_up = 1.0 / keep_prob

    for projection, _, _, _ in PROJECTIONS:
        shared_weight = shared_fp16[projection].float()
        mask = (torch.rand(shared_weight.shape, generator=generator) < keep_prob).to(torch.float32)
        upcycled = shared_weight * mask * scale_up
        ternary, q_scale = quantize_absmean(upcycled)
        tensors[f"{projection}_ternary"] = ternary
        tensors[f"{projection}_scale"] = q_scale

    return tensors


def main() -> None:
    logger = TeeLogger(EXPERIMENT_V2_LOG_PATH, mode="w")
    try:
        model_dir = resolve_model_dir(MODEL_REF)
        config = load_config(model_dir)
        tensor_index = build_tensor_index(model_dir)
        shared_fp16 = load_shared_fp16(tensor_index, layer_idx=LAYER_IDX)

        logger.emit("Create diverse experts")
        logger.emit(f"model_ref={MODEL_REF}")
        logger.emit(f"model_dir={model_dir}")
        logger.emit(f"layer_idx={LAYER_IDX}")
        logger.emit(f"drop_rate={DROP_RATE}")
        logger.emit(f"n_experts={N_EXPERTS}")
        logger.emit(f"checkpoint_n_experts={config.get('n_experts')}")
        logger.emit("")

        shared_q = {}
        for projection, _, _, _ in PROJECTIONS:
            ternary, q_scale = quantize_absmean(shared_fp16[projection].float())
            shared_q[f"{projection}_ternary"] = ternary
            shared_q[f"{projection}_scale"] = q_scale
        save_safetensors_file(
            shared_quantized_path(),
            shared_q,
            metadata={"model_ref": MODEL_REF, "layer_idx": str(LAYER_IDX), "kind": "shared_ternary"},
        )

        DIVERSE_EXPERTS_DIR.mkdir(parents=True, exist_ok=True)

        experts: dict[int, dict[str, torch.Tensor]] = {}
        for expert_idx in range(N_EXPERTS):
            tensors = create_diverse_expert_tensors(shared_fp16, expert_idx)
            experts[expert_idx] = tensors
            save_safetensors_file(
                diverse_expert_path(expert_idx),
                tensors,
                metadata={
                    "model_ref": MODEL_REF,
                    "layer_idx": str(LAYER_IDX),
                    "expert_idx": str(expert_idx),
                    "seed": str(LAYER_IDX * 1000 + expert_idx),
                    "drop_rate": str(DROP_RATE),
                },
            )

        logger.emit("Pairwise cosine similarities across all 8 synthesized experts")
        pairwise_rows = []
        offdiag = []
        for expert_i in range(N_EXPERTS):
            for expert_j in range(expert_i + 1, N_EXPERTS):
                cosine = expert_cosine_similarity(experts[expert_i], experts[expert_j])
                offdiag.append(cosine)
                pairwise_rows.append(
                    {
                        "expert_i": expert_i,
                        "expert_j": expert_j,
                        "cosine_similarity": cosine,
                    }
                )
                logger.emit(f"expert_{expert_i} vs expert_{expert_j}: cosine={cosine:.6f}")
        logger.emit("")
        logger.emit(
            f"average_pairwise_cosine={sum(offdiag) / len(offdiag):.6f}"
        )
        logger.emit(
            f"min_pairwise_cosine={min(offdiag):.6f} max_pairwise_cosine={max(offdiag):.6f}"
        )

        metadata_path = DIVERSE_EXPERTS_DIR / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "model_ref": MODEL_REF,
                    "layer_idx": LAYER_IDX,
                    "drop_rate": DROP_RATE,
                    "n_experts": N_EXPERTS,
                    "average_pairwise_cosine": sum(offdiag) / len(offdiag),
                    "min_pairwise_cosine": min(offdiag),
                    "max_pairwise_cosine": max(offdiag),
                    "pairwise_rows": pairwise_rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.emit(f"saved_dir={DIVERSE_EXPERTS_DIR}")
        logger.emit(f"metadata_path={metadata_path}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
