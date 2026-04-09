#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.paging import load_hybrid_paged_qwen
from outlier_engine.paging_v2 import load_hybrid_paged_qwen_v2
from outlier_engine.tokenizer import load_tokenizer


def _default_model() -> str:
    local = ROOT / "checkpoints" / "outlier-10b-v3" / "v3_checkpoints"
    if local.exists():
        return str(local)
    return "Outlier-Ai/Outlier-10B"


def _default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 ** 3)
    return usage / (1024 ** 2)


def _sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_generation(model, tokenizer, prompt: str, tokens: int, device_name: str) -> dict[str, Any]:
    prompt_text = tokenizer.prepare_prompt(prompt) if hasattr(tokenizer, "prepare_prompt") else prompt
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        raise SystemExit("Prompt encoded to an empty token list.")

    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device_name)
    attention_mask = torch.ones_like(generated, device=generated.device)
    past_key_values = None

    _sync(device_name)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(tokens):
            step_input = generated if past_key_values is None else generated[:, -1:]
            outputs = model(
                input_ids=step_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token = int(logits[0, -1].argmax().item())
            generated = torch.cat(
                [generated, torch.tensor([[next_token]], dtype=torch.long, device=generated.device)],
                dim=1,
            )
            attention_mask = torch.ones_like(generated, device=generated.device)
            past_key_values = getattr(outputs, "past_key_values", None)
    _sync(device_name)
    elapsed = time.perf_counter() - start

    stats = model.cache_stats() if hasattr(model, "cache_stats") else {}
    return {
        "tok_s": (tokens / elapsed) if elapsed > 0 else 0.0,
        "elapsed_s": elapsed,
        "peak_rss_gb": _peak_rss_gb(),
        "cache_hit_rate": float(stats.get("hit_rate", 0.0)),
        "prefetch_accuracy": float(stats.get("prefetch_accuracy", 0.0)),
        "stats": stats,
    }


def _load_impl(
    impl: str,
    model_path: str,
    device: str,
    max_experts: int,
    max_warm_cache: int,
    packed_experts_dir: str,
    monolith_path: str,
):
    common = dict(
        model_path=model_path,
        device=device,
        max_experts_in_memory=max_experts,
        max_warm_cache=max_warm_cache,
        packed_experts_dir=packed_experts_dir,
    )
    if impl == "baseline":
        model = load_hybrid_paged_qwen(**common)
    elif impl == "v2":
        model = load_hybrid_paged_qwen_v2(**common, monolith_path=monolith_path)
    else:
        raise ValueError(f"Unknown impl: {impl}")
    if hasattr(model, "enable_expert_prefetch"):
        model.enable_expert_prefetch()
    return model


def _child_main(args: argparse.Namespace) -> int:
    model = _load_impl(
        args.impl,
        args.model,
        args.device,
        args.max_experts,
        args.max_warm_cache,
        args.packed_experts_dir,
        args.monolith_path,
    )
    tokenizer = load_tokenizer(args.model)
    result = _run_generation(model, tokenizer, args.prompt, args.tokens, args.device)
    result["impl"] = args.impl
    result["model"] = args.model
    print(
        f"{args.impl}: tok/s={result['tok_s']:.2f} peak_rss_gb={result['peak_rss_gb']:.2f} "
        f"cache_hit_rate={result['cache_hit_rate']:.1%} prefetch_accuracy={result['prefetch_accuracy']:.1%}",
        flush=True,
    )
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)
    return 0


def _run_subprocess(args: argparse.Namespace, impl: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--impl",
        impl,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.tokens),
        "--device",
        args.device,
        "--max-experts",
        str(args.max_experts),
        "--max-warm-cache",
        str(args.max_warm_cache),
        "--packed-experts-dir",
        args.packed_experts_dir,
        "--monolith-path",
        args.monolith_path,
    ]
    env = os.environ.copy()
    env.setdefault("OUTLIER_BATCHED", "1")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT_JSON="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"Missing RESULT_JSON output for {impl}")


def _print_summary(baseline: dict[str, Any], v2: dict[str, Any]) -> None:
    speedup = v2["tok_s"] / baseline["tok_s"] if baseline["tok_s"] > 0 else 0.0
    print("comparison")
    print(
        f"baseline: tok/s={baseline['tok_s']:.2f} peak_rss_gb={baseline['peak_rss_gb']:.2f} "
        f"cache_hit_rate={baseline['cache_hit_rate']:.1%}"
    )
    print(
        f"paged_v2: tok/s={v2['tok_s']:.2f} peak_rss_gb={v2['peak_rss_gb']:.2f} "
        f"cache_hit_rate={v2['cache_hit_rate']:.1%}"
    )
    print(f"speedup_vs_baseline={speedup:.2f}x")
    print(f"prefetch_accuracy_baseline={baseline['prefetch_accuracy']:.1%}")
    print(f"prefetch_accuracy_v2={v2['prefetch_accuracy']:.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark paging.py vs paging_v2.py on 20-token decode.")
    parser.add_argument("--model", default=_default_model())
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--max-experts", type=int, default=8, dest="max_experts")
    parser.add_argument("--max-warm-cache", type=int, default=32, dest="max_warm_cache")
    parser.add_argument("--packed-experts-dir", default=str(ROOT / "packed_experts"))
    parser.add_argument("--monolith-path", default=str(ROOT / "experiments" / "monolith" / "experts.bin"))
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--impl", choices=("baseline", "v2"))
    args = parser.parse_args()

    if args.child:
        return _child_main(args)

    baseline = _run_subprocess(args, "baseline")
    v2 = _run_subprocess(args, "v2")
    _print_summary(baseline, v2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
