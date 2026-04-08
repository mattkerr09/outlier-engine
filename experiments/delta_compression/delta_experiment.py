from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from outlier_engine.paging import pack_ternary_tq10


MODEL_REF = "Outlier-Ai/Outlier-10B"
SAMPLED_LAYERS = (0, 8, 16)
CHUNK_SIZE = 4_000_000
PROJECTIONS = (
    ("gate", "gate_W", "gate_ternary", "gate_scale"),
    ("up", "up_W", "up_ternary", "up_scale"),
    ("down", "down_W", "down_ternary", "down_scale"),
)

OUTPUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUTPUT_DIR / "experiment_output.log"
RESULTS_PATH = OUTPUT_DIR / "results.md"


@dataclass
class ProjectionResult:
    name: str
    elements: int
    delta_scale: float
    sparsity: float
    full_tq10_bytes: int
    delta_tq10_bytes: int


@dataclass
class ExpertResult:
    layer_idx: int
    expert_idx: int
    elements: int
    sparsity: float
    reconstruction_error: float
    cosine_similarity: float
    full_tq10_bytes: int
    delta_tq10_bytes: int
    compression_ratio: float
    rle_tq10_bytes: int | None
    rle_compression_ratio: float | None
    projections: list[ProjectionResult]


class TeeLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def emit(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def bytes_to_mib(nbytes: int) -> float:
    return nbytes / (1024**2)


def tq10_nbytes(numel: int) -> int:
    return math.ceil(numel / 5)


def float16_nbytes(numel: int) -> int:
    return numel * torch.tensor([], dtype=torch.float16).element_size()


def build_tensor_shard_index(model_dir: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if (
                    key.startswith("base.model.layers.")
                    and ".mlp.shared_expert." in key
                ) or (
                    key.startswith("base.model.layers.")
                    and ".mlp.experts." in key
                ):
                    index[key] = str(shard)
    return index


def load_tensor(index: Dict[str, str], key: str) -> torch.Tensor:
    shard = index.get(key)
    if shard is None:
        raise KeyError(f"Tensor key not found in shard index: {key}")
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def chunk_ranges(total: int, chunk_size: int = CHUNK_SIZE) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def expert_prefix(layer_idx: int, expert_idx: int) -> str:
    return f"base.model.layers.{layer_idx}.mlp.experts.{expert_idx}"


def shared_prefix(layer_idx: int) -> str:
    return f"base.model.layers.{layer_idx}.mlp.shared_expert"


def compute_delta_scale(
    shared_weight: torch.Tensor,
    expert_ternary: torch.Tensor,
    expert_scale: float,
) -> float:
    shared_flat = shared_weight.reshape(-1)
    expert_flat = expert_ternary.reshape(-1)
    delta_abs_sum = 0.0

    for start, end in chunk_ranges(shared_flat.numel()):
        shared_chunk = shared_flat[start:end].float()
        routed_chunk = expert_flat[start:end].float().mul_(expert_scale)
        delta_abs_sum += (routed_chunk - shared_chunk).abs().sum().item()

    return max(delta_abs_sum / shared_flat.numel(), 1e-8)


def quantize_delta_chunk(delta_chunk: torch.Tensor, delta_scale: float) -> torch.Tensor:
    return (delta_chunk / delta_scale).round().clamp_(-1, 1).to(torch.int8)


def byte_rle_nbytes(packed: torch.Tensor) -> int:
    flat = packed.reshape(-1).to(torch.uint8).cpu()
    if flat.numel() == 0:
        return 0
    if flat.numel() == 1:
        return 2

    boundaries = torch.nonzero(flat[1:] != flat[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.tensor([0], dtype=torch.int64), boundaries])
    ends = torch.cat([boundaries, torch.tensor([flat.numel()], dtype=torch.int64)])
    run_lengths = ends - starts
    segments = ((run_lengths - 1) // 255 + 1).sum().item()
    return int(segments * 2)


def compute_projection_rle_size(
    shared_weight: torch.Tensor,
    expert_ternary: torch.Tensor,
    expert_scale: float,
    delta_scale: float,
    quant_scale_numel: int,
) -> int:
    shared_flat = shared_weight.reshape(-1)
    expert_flat = expert_ternary.reshape(-1)
    q_chunks: list[torch.Tensor] = []

    for start, end in chunk_ranges(shared_flat.numel()):
        shared_chunk = shared_flat[start:end].float()
        routed_chunk = expert_flat[start:end].float().mul_(expert_scale)
        delta_chunk = routed_chunk - shared_chunk
        q_chunks.append(quantize_delta_chunk(delta_chunk, delta_scale))

    quantized = torch.cat(q_chunks)
    packed = pack_ternary_tq10(quantized)
    return byte_rle_nbytes(packed) + float16_nbytes(quant_scale_numel)


def experts_are_identical(
    layer_idx: int,
    expert_a: int,
    expert_b: int,
    tensor_index: Dict[str, str],
) -> bool:
    for _, _, ternary_name, scale_name in PROJECTIONS:
        key_a = f"{expert_prefix(layer_idx, expert_a)}.{ternary_name}"
        key_b = f"{expert_prefix(layer_idx, expert_b)}.{ternary_name}"
        scale_a = f"{expert_prefix(layer_idx, expert_a)}.{scale_name}"
        scale_b = f"{expert_prefix(layer_idx, expert_b)}.{scale_name}"
        if not torch.equal(load_tensor(tensor_index, key_a), load_tensor(tensor_index, key_b)):
            return False
        if not torch.equal(load_tensor(tensor_index, scale_a), load_tensor(tensor_index, scale_b)):
            return False
    return True


def summarize_layer_uniqueness(
    layer_idx: int,
    n_experts: int,
    tensor_index: Dict[str, str],
) -> str:
    representatives: list[int] = []
    for expert_idx in range(n_experts):
        matched = False
        for rep_idx in representatives:
            if experts_are_identical(layer_idx, expert_idx, rep_idx, tensor_index):
                matched = True
                break
        if not matched:
            representatives.append(expert_idx)
    return (
        f"layer {layer_idx}: {len(representatives)} unique routed expert payloads "
        f"across {n_experts} experts"
    )


def evaluate_expert(
    layer_idx: int,
    expert_idx: int,
    tensor_index: Dict[str, str],
) -> ExpertResult:
    total_elements = 0
    zero_count = 0
    error_sq_sum = 0.0
    orig_sq_sum = 0.0
    recon_sq_sum = 0.0
    dot_sum = 0.0
    full_tq10_bytes = 0
    delta_tq10_bytes = 0
    projection_results: list[ProjectionResult] = []

    for projection_name, shared_name, ternary_name, scale_name in PROJECTIONS:
        shared_key = f"{shared_prefix(layer_idx)}.{shared_name}"
        expert_key = f"{expert_prefix(layer_idx, expert_idx)}.{ternary_name}"
        scale_key = f"{expert_prefix(layer_idx, expert_idx)}.{scale_name}"

        shared_weight = load_tensor(tensor_index, shared_key)
        expert_ternary = load_tensor(tensor_index, expert_key)
        expert_scale_tensor = load_tensor(tensor_index, scale_key)
        expert_scale = float(expert_scale_tensor.reshape(-1)[0].item())

        if shared_weight.shape != expert_ternary.shape:
            raise ValueError(
                f"Shape mismatch for layer {layer_idx} expert {expert_idx} "
                f"{projection_name}: shared={tuple(shared_weight.shape)} "
                f"expert={tuple(expert_ternary.shape)}"
            )

        projection_elements = shared_weight.numel()
        quant_scale_numel = expert_scale_tensor.numel()
        projection_full_tq10_bytes = tq10_nbytes(projection_elements) + float16_nbytes(
            quant_scale_numel
        )
        projection_delta_tq10_bytes = tq10_nbytes(projection_elements) + float16_nbytes(
            quant_scale_numel
        )
        delta_scale = compute_delta_scale(shared_weight, expert_ternary, expert_scale)

        shared_flat = shared_weight.reshape(-1)
        expert_flat = expert_ternary.reshape(-1)
        projection_zero_count = 0

        for start, end in chunk_ranges(projection_elements):
            shared_chunk = shared_flat[start:end].float()
            routed_chunk = expert_flat[start:end].float().mul_(expert_scale)
            delta_chunk = routed_chunk - shared_chunk
            quantized_chunk = quantize_delta_chunk(delta_chunk, delta_scale)
            quantized_float = quantized_chunk.float().mul_(delta_scale)
            recon_chunk = shared_chunk + quantized_float
            error_chunk = delta_chunk - quantized_float

            projection_zero_count += int((quantized_chunk == 0).sum().item())
            error_sq_sum += error_chunk.square().sum().item()
            orig_sq_sum += routed_chunk.square().sum().item()
            recon_sq_sum += recon_chunk.square().sum().item()
            dot_sum += (routed_chunk * recon_chunk).sum().item()

        total_elements += projection_elements
        zero_count += projection_zero_count
        full_tq10_bytes += projection_full_tq10_bytes
        delta_tq10_bytes += projection_delta_tq10_bytes
        projection_results.append(
            ProjectionResult(
                name=projection_name,
                elements=projection_elements,
                delta_scale=delta_scale,
                sparsity=projection_zero_count / projection_elements,
                full_tq10_bytes=projection_full_tq10_bytes,
                delta_tq10_bytes=projection_delta_tq10_bytes,
            )
        )

    reconstruction_error = math.sqrt(error_sq_sum)
    cosine_similarity = dot_sum / max(
        math.sqrt(orig_sq_sum) * math.sqrt(recon_sq_sum),
        1e-12,
    )
    sparsity = zero_count / total_elements
    compression_ratio = full_tq10_bytes / delta_tq10_bytes

    rle_tq10_bytes: int | None = None
    rle_compression_ratio: float | None = None
    if sparsity > 0.80:
        rle_total = 0
        for projection_name, shared_name, ternary_name, scale_name in PROJECTIONS:
            shared_key = f"{shared_prefix(layer_idx)}.{shared_name}"
            expert_key = f"{expert_prefix(layer_idx, expert_idx)}.{ternary_name}"
            scale_key = f"{expert_prefix(layer_idx, expert_idx)}.{scale_name}"
            projection_meta = next(
                item for item in projection_results if item.name == projection_name
            )
            shared_weight = load_tensor(tensor_index, shared_key)
            expert_ternary = load_tensor(tensor_index, expert_key)
            expert_scale_tensor = load_tensor(tensor_index, scale_key)
            expert_scale = float(expert_scale_tensor.reshape(-1)[0].item())
            rle_total += compute_projection_rle_size(
                shared_weight=shared_weight,
                expert_ternary=expert_ternary,
                expert_scale=expert_scale,
                delta_scale=projection_meta.delta_scale,
                quant_scale_numel=expert_scale_tensor.numel(),
            )
        rle_tq10_bytes = rle_total
        rle_compression_ratio = full_tq10_bytes / rle_tq10_bytes

    return ExpertResult(
        layer_idx=layer_idx,
        expert_idx=expert_idx,
        elements=total_elements,
        sparsity=sparsity,
        reconstruction_error=reconstruction_error,
        cosine_similarity=cosine_similarity,
        full_tq10_bytes=full_tq10_bytes,
        delta_tq10_bytes=delta_tq10_bytes,
        compression_ratio=compression_ratio,
        rle_tq10_bytes=rle_tq10_bytes,
        rle_compression_ratio=rle_compression_ratio,
        projections=projection_results,
    )


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_expert_line(result: ExpertResult) -> str:
    rle_part = "n/a"
    if result.rle_compression_ratio is not None and result.rle_tq10_bytes is not None:
        rle_part = (
            f"{bytes_to_mib(result.rle_tq10_bytes):.2f} MiB "
            f"({result.rle_compression_ratio:.3f}x)"
        )
    return (
        f"layer={result.layer_idx:02d} expert={result.expert_idx:02d} | "
        f"sparsity={result.sparsity * 100:6.2f}% | "
        f"recon_l2={result.reconstruction_error:.4f} | "
        f"cos={result.cosine_similarity:.6f} | "
        f"full_tq10={bytes_to_mib(result.full_tq10_bytes):7.2f} MiB | "
        f"delta_tq10={bytes_to_mib(result.delta_tq10_bytes):7.2f} MiB | "
        f"ratio={result.compression_ratio:.3f}x | "
        f"rle_tq10={rle_part}"
    )


def write_results_markdown(
    model_dir: Path,
    config: dict,
    results: list[ExpertResult],
    elapsed_s: float,
    uniqueness_notes: list[str],
) -> None:
    avg_sparsity = average([item.sparsity for item in results])
    avg_recon = average([item.reconstruction_error for item in results])
    avg_cos = average([item.cosine_similarity for item in results])
    avg_ratio = average([item.compression_ratio for item in results])

    rle_results = [item for item in results if item.rle_compression_ratio is not None]
    avg_rle_ratio = average([item.rle_compression_ratio for item in rle_results]) if rle_results else None
    sparse_fraction = len(rle_results) / len(results) if results else 0.0

    if avg_ratio <= 1.05 and (avg_rle_ratio is None or avg_rle_ratio < 1.25):
        recommendation = (
            "Not worth pursuing in its current storage format. "
            "Plain TQ1_0 delta packing does not beat full-expert TQ1_0, and the sparse "
            "RLE upside is too small to justify a patent non-provisional claim yet."
        )
    elif avg_cos >= 0.995 and avg_rle_ratio is not None and avg_rle_ratio >= 1.5:
        recommendation = (
            "Potentially worth a narrower follow-on. The value appears to come from "
            "sparsity-aware encoding on top of ternary deltas, not from plain TQ1_0 alone."
        )
    else:
        recommendation = (
            "Interesting research signal, but not strong enough yet for the "
            "non-provisional as a standalone compression claim. It needs either better "
            "reconstruction quality at higher sparsity or a more effective sparse codec."
        )

    lines = [
        "# Delta Compression Results",
        "",
        "## Summary",
        "",
        (
            f"Sampled `{len(results)}` routed experts from layers `{', '.join(map(str, SAMPLED_LAYERS))}` "
            f"of `{MODEL_REF}` from local snapshot `{model_dir}`."
        ),
        (
            "The experiment tested the hypothesis `expert ~= shared_ffn + ternary_delta`, "
            "where the delta was ternarized with scalar absmean quantization."
        ),
        "",
        f"- Runtime: `{elapsed_s:.1f}s`",
        f"- Average sparsity: `{avg_sparsity * 100:.2f}%`",
        f"- Average reconstruction error (L2): `{avg_recon:.4f}`",
        f"- Average cosine similarity: `{avg_cos:.6f}`",
        f"- Average compression ratio, full expert TQ1_0 / delta TQ1_0: `{avg_ratio:.3f}x`",
    ]

    if avg_rle_ratio is not None:
        lines.extend(
            [
                f"- Experts above 80% sparsity: `{len(rle_results)}/{len(results)}` ({sparse_fraction * 100:.1f}%)",
                f"- Average compression ratio with byte-level RLE over TQ1_0: `{avg_rle_ratio:.3f}x`",
            ]
        )
    else:
        lines.append("- Experts above 80% sparsity: `0`")

    if uniqueness_notes:
        lines.extend(
            [
                "",
                "## Checkpoint Notes",
                "",
                "Exact tensor comparison across gate/up/down ternary payloads and scales found:",
            ]
        )
        lines.extend([f"- {note}" for note in uniqueness_notes])

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "## Per-Expert Results",
            "",
            "| Layer | Expert | Sparsity | Recon L2 | Cosine | Full TQ1_0 MiB | Delta TQ1_0 MiB | Ratio | RLE TQ1_0 MiB | RLE Ratio |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for item in results:
        rle_bytes = f"{bytes_to_mib(item.rle_tq10_bytes):.2f}" if item.rle_tq10_bytes is not None else "n/a"
        rle_ratio = f"{item.rle_compression_ratio:.3f}x" if item.rle_compression_ratio is not None else "n/a"
        lines.append(
            "| "
            f"{item.layer_idx} | {item.expert_idx} | {item.sparsity * 100:.2f}% | "
            f"{item.reconstruction_error:.4f} | {item.cosine_similarity:.6f} | "
            f"{bytes_to_mib(item.full_tq10_bytes):.2f} | {bytes_to_mib(item.delta_tq10_bytes):.2f} | "
            f"{item.compression_ratio:.3f}x | {rle_bytes} | {rle_ratio} |"
        )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logger = TeeLogger(LOG_PATH)
    start_time = time.perf_counter()

    try:
        logger.emit("Delta compression experiment")
        logger.emit(f"model_ref={MODEL_REF}")
        logger.emit(f"sampled_layers={list(SAMPLED_LAYERS)}")
        logger.emit(f"log_path={LOG_PATH}")
        logger.emit("")

        model_dir = Path(
            snapshot_download(
                repo_id=MODEL_REF,
                allow_patterns=["config.json", "*.safetensors"],
            )
        )
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        tensor_index = build_tensor_shard_index(model_dir)
        n_experts = int(config["n_experts"])
        uniqueness_notes = [
            summarize_layer_uniqueness(layer_idx, n_experts, tensor_index)
            for layer_idx in SAMPLED_LAYERS
        ]

        logger.emit(f"model_dir={model_dir}")
        logger.emit(
            f"hidden_size={config['hidden_size']} intermediate_size={config['intermediate_size']} "
            f"layers={config['num_hidden_layers']} n_experts={n_experts}"
        )
        logger.emit("uniqueness_check:")
        for note in uniqueness_notes:
            logger.emit(f"  - {note}")
        logger.emit("")

        results: list[ExpertResult] = []
        total_targets = len(SAMPLED_LAYERS) * n_experts
        completed = 0

        for layer_idx in SAMPLED_LAYERS:
            logger.emit(f"[layer {layer_idx}] starting")
            for expert_idx in range(n_experts):
                expert_start = time.perf_counter()
                result = evaluate_expert(layer_idx, expert_idx, tensor_index)
                results.append(result)
                completed += 1
                elapsed = time.perf_counter() - expert_start
                logger.emit(
                    f"[{completed:02d}/{total_targets:02d}] "
                    f"{format_expert_line(result)} | elapsed={elapsed:.1f}s"
                )
            logger.emit("")

        total_elapsed = time.perf_counter() - start_time
        avg_sparsity = average([item.sparsity for item in results])
        avg_recon = average([item.reconstruction_error for item in results])
        avg_cos = average([item.cosine_similarity for item in results])
        avg_ratio = average([item.compression_ratio for item in results])
        rle_results = [item for item in results if item.rle_compression_ratio is not None]
        avg_rle_ratio = average([item.rle_compression_ratio for item in rle_results]) if rle_results else None

        logger.emit("Averages")
        logger.emit(f"experts={len(results)}")
        logger.emit(f"avg_sparsity={avg_sparsity * 100:.4f}%")
        logger.emit(f"avg_reconstruction_error={avg_recon:.6f}")
        logger.emit(f"avg_cosine_similarity={avg_cos:.8f}")
        logger.emit(f"avg_compression_ratio={avg_ratio:.6f}x")
        if avg_rle_ratio is not None:
            logger.emit(f"avg_rle_compression_ratio={avg_rle_ratio:.6f}x")
            logger.emit(f"rle_eligible_experts={len(rle_results)}")
        else:
            logger.emit("avg_rle_compression_ratio=n/a")
            logger.emit("rle_eligible_experts=0")
        logger.emit(f"total_elapsed={total_elapsed:.1f}s")

        write_results_markdown(
            model_dir=model_dir,
            config=config,
            results=results,
            elapsed_s=total_elapsed,
            uniqueness_notes=uniqueness_notes,
        )
        logger.emit(f"results_path={RESULTS_PATH}")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
