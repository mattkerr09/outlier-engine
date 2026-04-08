#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.et_routing import ETRouter
from outlier_engine.loader import load_model

OUTPUT_DIR = ROOT / "experiments" / "et_routing"
BENCH_LOG = OUTPUT_DIR / "bench_output.log"
RESULTS_MD = OUTPUT_DIR / "results.md"


def _default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _long_prompt() -> str:
    chunk = (
        "Answer briefly. Physics asks why gravity bends light. "
        "History asks who wrote the Federalist Papers. "
        "Biology asks how cells make ATP. "
        "Math asks why eigenvalues matter. "
    )
    return chunk * 24


def _capture_router_logits(loaded, prompt: str, sample_tokens: int) -> dict[int, torch.Tensor]:
    tokenizer = loaded.tokenizer
    prompt_text = tokenizer.prepare_prompt(prompt) if hasattr(tokenizer, "prepare_prompt") else prompt
    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) < sample_tokens:
        raise RuntimeError(f"Prompt only produced {len(prompt_ids)} tokens; need at least {sample_tokens}.")

    device = getattr(loaded.model, "outlier_device", loaded.device)
    if isinstance(device, torch.device):
        device_name = device.type
    else:
        device_name = str(device)
    input_ids = torch.tensor([prompt_ids[:sample_tokens]], dtype=torch.long, device=device_name)

    captured: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx, layer in enumerate(loaded.model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "router_weight"):
            continue

        def _hook(module, inputs, current_layer=layer_idx):
            hidden = inputs[0].detach()
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            logits = F.linear(hidden_flat.float(), module.router_weight.float()).cpu()
            if logits.shape[-1] > 0:
                captured[current_layer] = logits[:sample_tokens].clone()

        hooks.append(mlp.register_forward_pre_hook(_hook))

    with torch.no_grad():
        loaded.model(input_ids=input_ids, use_cache=False)

    for hook in hooks:
        hook.remove()
    return captured


def _top2_route(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_w, top_idx = torch.topk(probs, k=2, dim=-1)
    top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    return top_idx, top_w


def _evaluate_router(logits_by_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    total_tokens = 0
    total_experts = 0
    distribution: Counter[int] = Counter()
    agreement_sum = 0.0
    layer_summaries: list[dict[str, Any]] = []

    for layer_idx in sorted(logits_by_layer):
        logits = logits_by_layer[layer_idx]
        router = ETRouter(
            n_experts=logits.shape[-1],
            top_k_fallback=2,
            alpha=0.99,
            min_experts=1,
            max_experts=4,
        )
        router.calibrate(logits[: min(100, logits.shape[0])])
        et_idx, _et_w = router.route(logits)
        top2_idx, _top2_w = _top2_route(logits)

        layer_counts: Counter[int] = Counter()
        layer_agreement = 0.0
        for token_idx in range(et_idx.shape[0]):
            et_selected = {int(x) for x in et_idx[token_idx].tolist() if int(x) >= 0}
            top2_selected = {int(x) for x in top2_idx[token_idx].tolist()}
            layer_counts[len(et_selected)] += 1
            total_tokens += 1
            total_experts += len(et_selected)
            distribution[len(et_selected)] += 1
            layer_agreement += len(et_selected & top2_selected) / 2.0

        agreement_sum += layer_agreement
        layer_summaries.append(
            {
                "layer": layer_idx,
                "tokens": int(logits.shape[0]),
                "avg_experts_per_token": router.stats["avg_experts_per_token"],
                "agreement_with_top2": layer_agreement / max(int(logits.shape[0]), 1),
                "distribution": dict(sorted(layer_counts.items())),
            }
        )

    distribution_pct = {
        expert_count: count / max(total_tokens, 1)
        for expert_count, count in sorted(distribution.items())
    }
    return {
        "sampled_layers": len(layer_summaries),
        "sampled_tokens_total": total_tokens,
        "et_avg_experts_per_token": total_experts / max(total_tokens, 1),
        "top2_avg_experts_per_token": 2.0,
        "distribution_pct": distribution_pct,
        "agreement_with_top2": agreement_sum / max(total_tokens, 1),
        "layers": layer_summaries,
    }


def _benchmark_routing_time(logits_by_layer: dict[int, torch.Tensor], iterations: int) -> dict[str, float]:
    batches = [logits for _layer, logits in sorted(logits_by_layer.items())]
    total_tokens = sum(int(batch.shape[0]) for batch in batches) * iterations

    t0 = time.perf_counter()
    for _ in range(iterations):
        for logits in batches:
            probs = torch.softmax(logits.float(), dim=-1)
            torch.topk(probs, k=2, dim=-1)
    top2_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        for logits in batches:
            router = ETRouter(
                n_experts=logits.shape[-1],
                top_k_fallback=2,
                alpha=0.99,
                min_experts=1,
                max_experts=4,
            )
            router.calibrate(logits[: min(100, logits.shape[0])])
            router.route(logits)
    et_s = time.perf_counter() - t0

    return {
        "top2_total_s": top2_s,
        "et_total_s": et_s,
        "top2_us_per_token": (top2_s * 1e6) / max(total_tokens, 1),
        "et_us_per_token": (et_s * 1e6) / max(total_tokens, 1),
    }


def _write_results_md(summary: dict[str, Any], timing: dict[str, float], *, note: str = "") -> None:
    distribution_lines = "\n".join(
        f"- {expert_count} experts: {pct:.1%}"
        for expert_count, pct in summary["distribution_pct"].items()
    )
    replace_default = (
        "Not yet."
        if timing["et_us_per_token"] > timing["top2_us_per_token"] * 1.2
        else "Yes, if downstream quality checks confirm the expected MMLU gain."
    )
    note_block = f"\n## Benchmark Note\n\n{note}\n" if note else ""
    body = f"""# ET Routing Results

## Summary

- ET average experts per token: {summary['et_avg_experts_per_token']:.2f}
- Fixed top-2 experts per token: {summary['top2_avg_experts_per_token']:.2f}
- Agreement with top-2: {summary['agreement_with_top2']:.1%}
- Top-2 routing time: {timing['top2_us_per_token']:.2f} us/token
- ET routing time: {timing['et_us_per_token']:.2f} us/token

## ET Expert Count Distribution

{distribution_lines}

## Recommendation

{replace_default}
ET is selecting a variable number of experts while preserving a measurable overlap with fixed top-2. The current benchmark should be treated as routing-only evidence rather than a quality eval, so production replacement still depends on end-to-end accuracy validation.
{note_block}"""
    RESULTS_MD.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ET routing against fixed top-2 routing.")
    parser.add_argument("--model", default="Outlier-Ai/Outlier-10B-V2")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--sample-tokens", type=int, default=50)
    parser.add_argument("--timing-iterations", type=int, default=50)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    effective_model = args.model
    loaded = load_model(args.model, paged=True, device=args.device)
    logits_by_layer = _capture_router_logits(loaded, _long_prompt(), args.sample_tokens)
    fallback_note = ""
    if not logits_by_layer and args.model.endswith("-V2"):
        effective_model = "Outlier-Ai/Outlier-10B"
        fallback_note = (
            f"{args.model} exposed no MoE router layers in this checkout; "
            f"routing benchmark was run against {effective_model} instead."
        )
        loaded = load_model(effective_model, paged=True, device=args.device)
        logits_by_layer = _capture_router_logits(loaded, _long_prompt(), args.sample_tokens)
    summary = _evaluate_router(logits_by_layer)
    timing = _benchmark_routing_time(logits_by_layer, args.timing_iterations)

    payload = {
        "requested_model": args.model,
        "effective_model": effective_model,
        "device": args.device,
        "sample_tokens_per_layer": args.sample_tokens,
        "note": fallback_note,
        "summary": summary,
        "timing": timing,
    }
    BENCH_LOG.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_results_md(summary, timing, note=fallback_note)

    print(f"requested_model={args.model}")
    print(f"effective_model={effective_model}")
    print(f"device={args.device}")
    if fallback_note:
        print(f"note={fallback_note}")
    print(f"sample_tokens_per_layer={args.sample_tokens}")
    print(f"sampled_layers={summary['sampled_layers']}")
    print(f"sampled_tokens_total={summary['sampled_tokens_total']}")
    print(f"et_avg_experts_per_token={summary['et_avg_experts_per_token']:.2f}")
    print(f"top2_avg_experts_per_token={summary['top2_avg_experts_per_token']:.2f}")
    print(f"agreement_with_top2={summary['agreement_with_top2']:.1%}")
    print(f"distribution_pct={summary['distribution_pct']}")
    print(f"top2_us_per_token={timing['top2_us_per_token']:.2f}")
    print(f"et_us_per_token={timing['et_us_per_token']:.2f}")
    print(f"bench_output={BENCH_LOG}")
    print(f"results_md={RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
