from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Optional

import torch

from .generate import benchmark_generation, stream_generate
from .loader import inspect_model, load_model
from .paging import _default_packed_dir, repack_ternary_experts

DEFAULT_MODEL = "Outlier-Ai/Outlier-10B"
ENGINE_VERSION = "0.3"

BENCH_PROMPTS = [
    ("short", "Hello"),
    ("medium", "Explain quantum computing in simple terms"),
    (
        "long",
        "Write a detailed analysis of the economic impacts of artificial intelligence on the labor market over the next decade",
    ),
]

DEMO_PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in one sentence:",
    "Write a haiku about artificial intelligence:",
]


def _cache_summary_lines(cache_stats: Optional[dict]) -> list[str]:
    if not cache_stats:
        return []
    lines = [
        "Expert cache: "
        f"hot={cache_stats.get('hot_hits', 0)} "
        f"warm={cache_stats.get('warm_hits', 0)} "
        f"cold={cache_stats.get('cold_misses', cache_stats.get('misses', 0))} "
        f"/ {cache_stats.get('lookups', 0)} lookups "
        f"({cache_stats.get('hit_rate', 0.0):.1%} overall hit rate)"
    ]
    if "hot_cache_entries" in cache_stats:
        lines.append(
            "Hot cache: "
            f"{cache_stats.get('hot_cache_entries', 0)} entries, "
            f"{cache_stats.get('hot_cache_mb', 0.0):.1f} MB"
        )
    if cache_stats.get("disk_loads"):
        lines.append(
            "Disk loads: "
            f"{cache_stats['disk_loads']} total, "
            f"{cache_stats['avg_disk_load_ms']:.1f} ms average"
        )
    return lines


def _run_generation(
    loaded,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    verbose: bool,
) -> dict:
    generator = stream_generate(
        loaded,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        file=sys.stdout,
        verbose=verbose,
        verbose_file=sys.stderr,
    )
    t0 = time.perf_counter()
    token_count = 0
    cache_stats = None
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            result = stop.value or {}
            token_count = int(result.get("tokens", token_count))
            cache_stats = result.get("cache_stats")
            break
    elapsed = time.perf_counter() - t0
    tok_s = token_count / max(elapsed, 1e-6)
    return {
        "tokens": token_count,
        "elapsed_s": elapsed,
        "tokens_per_s": tok_s,
        "cache_stats": cache_stats,
    }


def _resolve_run_inputs(args) -> tuple[str, str]:
    values = getattr(args, "inputs", [])
    if not values:
        raise SystemExit("run requires a prompt")
    if len(values) == 1:
        return args.model or DEFAULT_MODEL, values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise SystemExit("run accepts either: run \"prompt\" or run MODEL \"prompt\"")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outlier-engine", description="Public inference engine for Outlier models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run autoregressive generation.")
    run_parser.add_argument("inputs", nargs="+")
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--hf-token", default=None, dest="hf_token")
    run_parser.add_argument("--paged", action="store_true")
    run_parser.add_argument("--full", action="store_true")
    run_parser.add_argument("--prefetch", action="store_true")
    run_parser.add_argument("--max-experts", type=int, default=64, dest="max_experts")
    run_parser.add_argument("--max-warm-cache", type=int, default=256, dest="max_warm_cache")
    run_parser.add_argument("--verbose", action="store_true", help="Print token IDs and repr() chunks while generating.")

    info_parser = subparsers.add_parser("info", help="Print model architecture and artifact info.")
    info_parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    info_parser.add_argument("--hf-token", default=None, dest="hf_token")

    bench_parser = subparsers.add_parser("bench", help="Run a quick throughput benchmark.")
    bench_parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    bench_parser.add_argument("--max-tokens", type=int, default=20, dest="max_tokens")
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--hf-token", default=None, dest="hf_token")
    bench_parser.add_argument("--paged", action="store_true")
    bench_parser.add_argument("--full", action="store_true")
    bench_parser.add_argument("--prefetch", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Run a three-prompt terminal demo.")
    demo_parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    demo_parser.add_argument("--max-tokens", type=int, default=20, dest="max_tokens")
    demo_parser.add_argument("--temperature", type=float, default=0.7)
    demo_parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    demo_parser.add_argument("--device", default=None)
    demo_parser.add_argument("--hf-token", default=None, dest="hf_token")
    demo_parser.add_argument("--paged", action="store_true")
    demo_parser.add_argument("--full", action="store_true")
    demo_parser.add_argument("--prefetch", action="store_true")
    demo_parser.add_argument("--max-experts", type=int, default=64, dest="max_experts")
    demo_parser.add_argument("--max-warm-cache", type=int, default=256, dest="max_warm_cache")

    repack_parser = subparsers.add_parser("repack", help="Repack legacy MoE experts into local TQ1_0 cache.")
    repack_parser.add_argument("model")
    repack_parser.add_argument("--output-dir", default=str(_default_packed_dir()), dest="output_dir")
    repack_parser.add_argument("--hf-token", default=None, dest="hf_token")

    return parser


def _resolve_paged_flag(args) -> Optional[bool]:
    if getattr(args, "paged", False):
        return True
    if getattr(args, "full", False):
        return False
    return None


def _resolve_token(args) -> Optional[str]:
    return args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _infer_total_params_str(model_ref: str, config: dict) -> str:
    total_params = config.get("total_params_B") or config.get("base_params_B") or config.get("n_params")
    if isinstance(total_params, (int, float)):
        return f"{float(total_params):.1f}B"
    match = re.search(r"(\d+(?:\.\d+)?)B", model_ref)
    if match:
        return f"{float(match.group(1)):.1f}B"
    return "unknown"


def _infer_active_params_str(model_ref: str, config: dict) -> str:
    n_experts = int(config.get("n_experts", 0) or 0)
    top_k = int(config.get("top_k", 0) or 0)
    if n_experts and top_k:
        return f"top-{top_k}/{n_experts} experts"
    return _infer_total_params_str(model_ref, config)


def _peak_memory_gb(device: Optional[str]) -> Optional[float]:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps" and getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            return None
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        info = inspect_model(args.model, token=_resolve_token(args))
        config = info["config"]
        total_params_str = _infer_total_params_str(info["resolved_model_ref"], config)
        active_params = _infer_active_params_str(info["resolved_model_ref"], config)
        print(
            f"{info['resolved_model_ref']} | params={total_params_str} | "
            f"active={active_params} | device=auto | engine=v{ENGINE_VERSION}"
        )
        return 0

    if args.command == "run":
        model_ref, prompt = _resolve_run_inputs(args)
        loaded = load_model(
            model_ref,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
            prefetch=args.prefetch,
            max_experts_in_memory=args.max_experts,
            max_warm_cache=args.max_warm_cache,
        )
        result = _run_generation(
            loaded,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
        )
        print()
        print(
            f"Generated {result['tokens']} tokens in {result['elapsed_s']:.1f}s "
            f"({result['tokens_per_s']:.1f} tok/s)"
        )
        for line in _cache_summary_lines(result.get("cache_stats")):
            print(line)
        return 0

    if args.command == "bench":
        load_t0 = time.perf_counter()
        loaded = load_model(
            args.model,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
            prefetch=args.prefetch,
        )
        load_time = time.perf_counter() - load_t0
        results = [
            benchmark_generation(
                loaded,
                prompt,
                max_tokens=args.max_tokens,
            )
            for _, prompt in BENCH_PROMPTS
        ]
        avg_tok_s = sum(result["tokens_per_s"] for result in results) / max(len(results), 1)
        peak_memory = _peak_memory_gb(getattr(loaded, "device", args.device))

        print("outlier-engine bench v0.2")
        print(f"Model: {args.model}")
        print(f"Device: {loaded.device}")
        for idx, ((label, _), result) in enumerate(zip(BENCH_PROMPTS, results), start=1):
            print(
                f"Prompt {idx} ({label}): {result['tokens']:>3} tokens in "
                f"{result['elapsed_s']:.1f}s ({result['tokens_per_s']:.1f} tok/s)"
            )
        print(f"Average: {avg_tok_s:.1f} tok/s")
        print(f"Load time: {load_time:.1f}s")
        if peak_memory is None:
            print("Peak memory: n/a")
        else:
            print(f"Peak memory: {peak_memory:.1f} GB")
        return 0

    if args.command == "demo":
        loaded = load_model(
            args.model,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
            prefetch=args.prefetch,
            max_experts_in_memory=args.max_experts,
            max_warm_cache=args.max_warm_cache,
        )
        results = []
        for idx, prompt in enumerate(DEMO_PROMPTS, start=1):
            print(f"Prompt {idx}: {prompt}")
            result = _run_generation(
                loaded,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                verbose=False,
            )
            print()
            print(
                f"Prompt {idx} done: {result['tokens']} tokens in {result['elapsed_s']:.1f}s "
                f"({result['tokens_per_s']:.1f} tok/s)"
            )
            for line in _cache_summary_lines(result.get("cache_stats")):
                print(line)
            print()
            results.append(result)

        avg_tok_s = sum(item["tokens_per_s"] for item in results) / max(len(results), 1)
        print("Demo Summary")
        print(f"Model: {args.model}")
        print(f"Device: {loaded.device}")
        print(f"Mode: {'paged' if _resolve_paged_flag(args) else 'standard'}")
        print(f"Avg tok/s across prompts: {avg_tok_s:.1f}")
        return 0

    if args.command == "repack":
        meta = repack_ternary_experts(
            args.model,
            output_dir=args.output_dir,
            token=_resolve_token(args),
        )
        print(f"Repacked experts for {args.model}")
        print(f"Output dir: {args.output_dir}")
        print(f"Ternary tensors: {meta['ternary_tensors']}")
        print(f"Scale tensors: {meta['scale_tensors']}")
        print(f"Original expert data: {meta['original_mb']:.1f} MB")
        print(f"Packed expert data: {meta['packed_mb']:.1f} MB")
        print(f"Compression ratio: {meta['compression_ratio']:.2f}x")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
