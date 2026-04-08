from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch

from .generate import benchmark_generation, stream_generate
from .loader import inspect_model, load_model
from .paging import _default_packed_dir, repack_ternary_experts

BENCH_PROMPTS = [
    ("short", "Hello"),
    ("medium", "Explain quantum computing in simple terms"),
    (
        "long",
        "Write a detailed analysis of the economic impacts of artificial intelligence on the labor market over the next decade",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="outlier-engine", description="Public inference engine for Outlier models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run autoregressive generation.")
    run_parser.add_argument("model")
    run_parser.add_argument("prompt")
    run_parser.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    run_parser.add_argument("--device", default=None)
    run_parser.add_argument("--hf-token", default=None, dest="hf_token")
    run_parser.add_argument("--paged", action="store_true")
    run_parser.add_argument("--full", action="store_true")
    run_parser.add_argument("--max-experts", type=int, default=64, dest="max_experts")
    run_parser.add_argument("--max-warm-cache", type=int, default=256, dest="max_warm_cache")
    run_parser.add_argument("--verbose", action="store_true", help="Print token IDs and repr() chunks while generating.")

    info_parser = subparsers.add_parser("info", help="Print model architecture and artifact info.")
    info_parser.add_argument("model")
    info_parser.add_argument("--hf-token", default=None, dest="hf_token")

    bench_parser = subparsers.add_parser("bench", help="Run a quick throughput benchmark.")
    bench_parser.add_argument("model")
    bench_parser.add_argument("--max-tokens", type=int, default=20, dest="max_tokens")
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--hf-token", default=None, dest="hf_token")
    bench_parser.add_argument("--paged", action="store_true")
    bench_parser.add_argument("--full", action="store_true")

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
        payload = {
            "model_ref": info["model_ref"],
            "model_dir": info["model_dir"],
            "n_layers": config.get("n_layers"),
            "hidden_dim": config.get("hidden_dim"),
            "intermediate_dim": config.get("intermediate_dim"),
            "n_heads": config.get("n_heads"),
            "n_experts": config.get("n_experts", 0),
            "top_k": config.get("top_k", 0),
            "vocab_size": config.get("vocab_size"),
            "max_seq_len": config.get("max_seq_len"),
            "artifact_index": info["artifacts"],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "run":
        loaded = load_model(
            args.model,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
            max_experts_in_memory=args.max_experts,
            max_warm_cache=args.max_warm_cache,
        )
        generator = stream_generate(
            loaded,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            file=sys.stdout,
            verbose=args.verbose,
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
        print()
        elapsed = time.perf_counter() - t0
        tok_s = token_count / max(elapsed, 1e-6)
        print(f"Generated {token_count} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)")
        if cache_stats:
            print(
                "Expert cache: "
                f"{cache_stats['hits']} hits / {cache_stats['lookups']} lookups "
                f"({cache_stats['hit_rate']:.1%} hit rate), "
                f"{cache_stats['evictions']} evictions"
            )
            if cache_stats.get("disk_loads"):
                print(
                    "Disk loads: "
                    f"{cache_stats['disk_loads']} total, "
                    f"{cache_stats['avg_disk_load_ms']:.1f} ms average"
                )
        return 0

    if args.command == "bench":
        load_t0 = time.perf_counter()
        loaded = load_model(
            args.model,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
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
