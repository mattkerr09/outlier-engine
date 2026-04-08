from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .generate import benchmark_generation, stream_generate
from .loader import inspect_model, load_model


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
    run_parser.add_argument("--max-experts", type=int, default=4, dest="max_experts")
    run_parser.add_argument("--max-warm-cache", type=int, default=16, dest="max_warm_cache")
    run_parser.add_argument("--verbose", action="store_true", help="Print token IDs and repr() chunks while generating.")

    info_parser = subparsers.add_parser("info", help="Print model architecture and artifact info.")
    info_parser.add_argument("model")
    info_parser.add_argument("--hf-token", default=None, dest="hf_token")

    bench_parser = subparsers.add_parser("bench", help="Run a quick throughput benchmark.")
    bench_parser.add_argument("model")
    bench_parser.add_argument("--prompt", default="Hello, world")
    bench_parser.add_argument("--max-tokens", type=int, default=16, dest="max_tokens")
    bench_parser.add_argument("--device", default=None)
    bench_parser.add_argument("--hf-token", default=None, dest="hf_token")
    bench_parser.add_argument("--paged", action="store_true")
    bench_parser.add_argument("--full", action="store_true")

    return parser


def _resolve_paged_flag(args) -> Optional[bool]:
    if getattr(args, "paged", False):
        return True
    if getattr(args, "full", False):
        return False
    return None


def _resolve_token(args) -> Optional[str]:
    return args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


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
        while True:
            try:
                next(generator)
            except StopIteration as stop:
                result = stop.value or {}
                token_count = int(result.get("tokens", token_count))
                break
        print()
        elapsed = time.perf_counter() - t0
        tok_s = token_count / max(elapsed, 1e-6)
        print(f"Generated {token_count} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)")
        return 0

    if args.command == "bench":
        loaded = load_model(
            args.model,
            token=_resolve_token(args),
            device=args.device,
            paged=_resolve_paged_flag(args),
        )
        metrics = benchmark_generation(
            loaded,
            args.prompt,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
