#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outlier_engine.loader import load_model


def _peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux typically reports KB.
    if usage > 1_000_000_000:
        return usage / (1024 ** 3)
    return usage / (1024 ** 2)


def _default_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(device: str) -> None:
    if device == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _format_cache(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "cache=unavailable"
    summary = (
        f"hot_hits={stats.get('hot_hits', 0)} "
        f"warm_hits={stats.get('warm_hits', 0)} "
        f"cold_misses={stats.get('cold_misses', stats.get('misses', 0))} "
        f"lookups={stats.get('lookups', 0)} "
        f"hit_rate={stats.get('hit_rate', 0.0):.1%}"
    )
    if "prefetch_accuracy" in stats:
        summary += f" prefetch_accuracy={stats.get('prefetch_accuracy', 0.0):.1%}"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark paged Outlier inference token by token.")
    parser.add_argument("--model", default="Outlier-Ai/Outlier-10B")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=5)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    print(f"model={args.model}")
    print(f"device={args.device}")
    print(f"prompt={args.prompt!r}")
    print(f"target_new_tokens={args.tokens}")
    print("loading paged model...", flush=True)

    load_t0 = time.perf_counter()
    loaded = load_model(args.model, paged=True, device=args.device, prefetch=args.prefetch)
    _sync(args.device)
    load_s = time.perf_counter() - load_t0

    tokenizer = loaded.tokenizer
    prompt_text = tokenizer.prepare_prompt(args.prompt) if hasattr(tokenizer, "prepare_prompt") else args.prompt
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        raise SystemExit("Prompt encoded to an empty token list.")

    device = getattr(loaded.model, "outlier_device", loaded.device)
    if isinstance(device, torch.device):
        device_name = device.type
    else:
        device_name = str(device)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device_name)

    print(f"backend={loaded.backend}")
    print(f"prompt_tokens={input_ids.shape[1]}")
    print(f"load_time_s={load_s:.2f}")
    print(f"peak_rss_gb_after_load={_peak_rss_gb():.2f}")
    print("starting token benchmark...", flush=True)

    token_latencies: list[float] = []
    token_texts: list[str] = []
    cache_snapshots: list[dict[str, Any]] = []
    past_key_values = None
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    generated = input_ids

    with torch.no_grad():
        for idx in range(args.tokens):
            if hasattr(loaded.model, "outlier_page_manager"):
                loaded.model.outlier_page_manager.debug_forward_start(f"token_{idx + 1}")
            step_input = generated if past_key_values is None else generated[:, -1:]
            step_mask = attention_mask
            t0 = time.perf_counter()
            outputs = loaded.model(
                input_ids=step_input,
                attention_mask=step_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            _sync(device_name)
            elapsed = time.perf_counter() - t0
            token_latencies.append(elapsed)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            next_token = int(logits[0, -1].argmax().item())
            token_text = tokenizer.decode([next_token]) if hasattr(tokenizer, "decode") else str(next_token)
            token_texts.append(token_text)
            generated = torch.cat(
                [generated, torch.tensor([[next_token]], dtype=torch.long, device=generated.device)],
                dim=1,
            )
            attention_mask = torch.ones_like(generated, device=generated.device)
            past_key_values = getattr(outputs, "past_key_values", None)

            stats = loaded.model.cache_stats() if hasattr(loaded.model, "cache_stats") else None
            cache_snapshots.append(dict(stats) if stats else {})
            print(
                f"token_{idx + 1}: latency_s={elapsed:.2f} "
                f"text={token_text!r} {_format_cache(stats)} "
                f"hot_cache_entries={stats.get('hot_cache_entries', 0) if stats else 0} "
                f"hot_cache_mb={stats.get('hot_cache_mb', 0.0):.1f} "
                f"peak_rss_gb={_peak_rss_gb():.2f}",
                flush=True,
            )

    token1 = token_latencies[0] if token_latencies else 0.0
    tail = token_latencies[1:] if len(token_latencies) > 1 else []
    tail_avg = (sum(tail) / len(tail)) if tail else 0.0
    tail_hot = token_latencies[2:] if len(token_latencies) > 2 else []
    tail_hot_avg = (sum(tail_hot) / len(tail_hot)) if tail_hot else 0.0
    post_first = cache_snapshots[0] if cache_snapshots else {}
    final_stats = cache_snapshots[-1] if cache_snapshots else {}
    lookups_after_first = int(final_stats.get("lookups", 0)) - int(post_first.get("lookups", 0))
    hot_hits_after_first = int(final_stats.get("hot_hits", 0)) - int(post_first.get("hot_hits", 0))
    warm_hits_after_first = int(final_stats.get("warm_hits", 0)) - int(post_first.get("warm_hits", 0))
    cold_after_first = int(final_stats.get("cold_misses", final_stats.get("misses", 0))) - int(
        post_first.get("cold_misses", post_first.get("misses", 0))
    )
    hot_hit_rate_after_first = hot_hits_after_first / lookups_after_first if lookups_after_first > 0 else 0.0
    warm_hit_rate_after_first = warm_hits_after_first / lookups_after_first if lookups_after_first > 0 else 0.0
    cold_miss_rate_after_first = cold_after_first / lookups_after_first if lookups_after_first > 0 else 0.0

    print("summary")
    print(f"token_1_latency_s={token1:.2f}")
    print(f"token_2_to_5_avg_latency_s={tail_avg:.2f}")
    print(f"token_3_to_5_avg_latency_s={tail_hot_avg:.2f}")
    print(f"hot_hit_rate_after_token_1={hot_hit_rate_after_first:.1%}")
    print(f"warm_hit_rate_after_token_1={warm_hit_rate_after_first:.1%}")
    print(f"cold_miss_rate_after_token_1={cold_miss_rate_after_first:.1%}")
    if "prefetch_accuracy" in final_stats:
        print(f"prefetch_accuracy={final_stats.get('prefetch_accuracy', 0.0):.1%}")
        print(f"prefetches_issued={int(final_stats.get('prefetches_issued', 0))}")
        print(f"prefetch_hits={int(final_stats.get('prefetch_hits', 0))}")
        print(f"prefetch_wastes={int(final_stats.get('prefetch_wastes', 0))}")
    print(f"hot_cache_entries={final_stats.get('hot_cache_entries', 0)}")
    print(f"hot_cache_mb={final_stats.get('hot_cache_mb', 0.0):.1f}")
    print(f"final_cache_stats={final_stats}")
    print(f"peak_rss_gb={_peak_rss_gb():.2f}")
    print(f"generated_text={''.join(token_texts)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
