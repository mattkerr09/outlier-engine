#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TRAIN_PATH = Path("~/outlier/data/train/combined.jsonl").expanduser()
DEFAULT_OUTPUT_DIR = Path("data/distillation")
TEACHER_REF = "Qwen/Qwen2.5-7B-Instruct"


def log(message: str) -> None:
    print(message, flush=True)


def current_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def peak_rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage > 1_000_000_000:
        return usage / (1024 ** 3)
    return usage / (1024 ** 2)


def disk_usage_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total / (1024 ** 3)


def detect_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "cpu"}:
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float16


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def count_domains(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[row.get("domain", "unknown")] += 1
    return counter


def compute_domain_quotas(counts: Counter[str], target: int, min_per_domain: int = 100) -> dict[str, int]:
    if target <= 0:
        return {domain: 0 for domain in counts}
    if target < len(counts):
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        quotas = {domain: 0 for domain in counts}
        for domain, _count in ordered[:target]:
            quotas[domain] = 1
        return quotas

    total = sum(counts.values())
    quotas: dict[str, int] = {}
    remaining = target
    leftovers: dict[str, int] = {}
    for domain, count in counts.items():
        quota = min(count, min_per_domain)
        quotas[domain] = quota
        remaining -= quota
        leftovers[domain] = max(count - quota, 0)

    if remaining <= 0:
        domains = list(quotas)
        idx = 0
        while sum(quotas.values()) > target:
            domain = domains[idx % len(domains)]
            if quotas[domain] > 1:
                quotas[domain] -= 1
            idx += 1
        return quotas

    leftover_total = sum(leftovers.values())
    if leftover_total <= 0:
        return quotas

    fractional: list[tuple[float, str]] = []
    for domain, count in leftovers.items():
        exact = remaining * (count / leftover_total)
        add = min(count, int(math.floor(exact)))
        quotas[domain] += add
        fractional.append((exact - add, domain))

    assigned = sum(quotas.values())
    extra = target - assigned
    for _, domain in sorted(fractional, reverse=True):
        if extra <= 0:
            break
        if quotas[domain] < counts[domain]:
            quotas[domain] += 1
            extra -= 1
    return quotas


def sample_stratified(rows: list[dict[str, Any]], target: int, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(row.get("domain", "unknown"), []).append(row)
    for items in by_domain.values():
        rng.shuffle(items)

    quotas = compute_domain_quotas(Counter({k: len(v) for k, v in by_domain.items()}), target)
    selected: list[dict[str, Any]] = []
    for domain, quota in quotas.items():
        selected.extend(by_domain[domain][:quota])
    rng.shuffle(selected)
    return selected[:target]


def row_to_messages(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if "messages" in row:
        prompt_preview = ""
        messages: list[dict[str, str]] = []
        for message in row["messages"]:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            messages.append({"role": role, "content": content})
            if role == "user" and not prompt_preview:
                prompt_preview = content
        completion = str(row.get("completion", "")).strip()
        if completion:
            messages.append({"role": "assistant", "content": completion})
        return messages, prompt_preview or completion[:80]

    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    return messages, prompt


def render_training_text(tokenizer, row: dict[str, Any]) -> tuple[str, str]:
    messages, preview = row_to_messages(row)
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
        text = "\n\n".join(parts)
    return text, preview


def save_shard(output_dir: Path, shard_id: int, items: list[dict[str, Any]]) -> Path:
    shard_path = output_dir / f"corpus_shard_{shard_id:04d}.pt"
    torch.save(items, shard_path)
    return shard_path


def load_random_examples(output_dir: Path, sample_count: int = 10, seed: int = 42) -> list[dict[str, Any]]:
    shard_paths = sorted(output_dir.glob("corpus_shard_*.pt"))
    rng = random.Random(seed)
    rng.shuffle(shard_paths)
    collected: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        items = torch.load(shard_path, map_location="cpu")
        rng.shuffle(items)
        for item in items:
            collected.append(item)
            if len(collected) >= sample_count:
                return collected
    return collected


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a top-k teacher-logit distillation corpus.")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_path = Path(args.train_file).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device(args.device)
    dtype = detect_dtype(device)
    log(f"train_file={train_path}")
    log(f"output_dir={output_dir}")
    log(f"teacher={TEACHER_REF}")
    log(f"device={device}")
    log(f"dtype={dtype}")

    rows = read_jsonl(train_path)
    domain_counts = count_domains(rows)
    log(f"examples_found={len(rows)}")
    log("format=jsonl")
    log(f"domain_counts={dict(domain_counts)}")

    target = min(args.max_examples, len(rows))
    selected = sample_stratified(rows, target=target, seed=args.seed)
    selected_domain_counts = count_domains(selected)
    log(f"selected_examples={len(selected)}")
    log(f"selected_domain_counts={dict(selected_domain_counts)}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_REF, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_REF,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    shard_items: list[dict[str, Any]] = []
    shard_paths: list[str] = []
    start = time.perf_counter()
    corpus_token_count = 0

    with torch.inference_mode():
        for idx, row in enumerate(selected, start=1):
            training_text, preview = render_training_text(tokenizer, row)
            encoded = tokenizer(
                training_text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            topk = torch.topk(logits, k=args.top_k, dim=-1)

            item = {
                "input_ids": encoded["input_ids"][0].to(torch.int32).cpu(),
                "attention_mask": encoded["attention_mask"][0].to(torch.int8).cpu(),
                "top_k_logits": topk.values.to(torch.float16).cpu(),
                "top_k_indices": topk.indices.to(torch.int32).cpu(),
                "domain": row.get("domain", "unknown"),
                "text_preview": preview[:120],
                "source_teacher": row.get("teacher", "unknown"),
            }
            shard_items.append(item)
            corpus_token_count += int(encoded["input_ids"].shape[1])

            del outputs, logits, topk, input_ids, attention_mask
            if device == "mps":
                torch.mps.synchronize()

            if len(shard_items) >= args.shard_size:
                shard_id = len(shard_paths)
                shard_path = save_shard(output_dir, shard_id, shard_items)
                shard_paths.append(str(shard_path))
                shard_items = []
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()

            if idx % 100 == 0 or idx == len(selected):
                elapsed = time.perf_counter() - start
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta_s = (len(selected) - idx) / rate if rate > 0 else 0.0
                log(
                    f"progress examples={idx}/{len(selected)} "
                    f"elapsed_min={elapsed/60.0:.1f} eta_min={eta_s/60.0:.1f} "
                    f"disk_gb={disk_usage_gb(output_dir):.2f} current_rss_gb={current_rss_gb():.2f} "
                    f"peak_rss_gb={peak_rss_gb():.2f}"
                )

    if shard_items:
        shard_id = len(shard_paths)
        shard_path = save_shard(output_dir, shard_id, shard_items)
        shard_paths.append(str(shard_path))

    metadata = {
        "teacher": TEACHER_REF,
        "train_file": str(train_path),
        "device": device,
        "dtype": str(dtype),
        "examples": len(selected),
        "top_k": args.top_k,
        "max_length": args.max_length,
        "shard_size": args.shard_size,
        "shards": shard_paths,
        "selected_domain_counts": dict(selected_domain_counts),
        "total_tokens": corpus_token_count,
        "disk_size_gb": disk_usage_gb(output_dir),
    }
    metadata_path = output_dir / "corpus_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    samples = load_random_examples(output_dir, sample_count=10, seed=args.seed)
    log("verification_samples")
    for i, sample in enumerate(samples, start=1):
        top_token = int(sample["top_k_indices"][0, 0].item())
        top_token_text = tokenizer.decode([top_token])
        has_nan = bool(torch.isnan(sample["top_k_logits"]).any().item())
        all_zero = bool((sample["top_k_logits"].abs().sum() == 0).item())
        log(
            f"sample_{i}: preview={sample['text_preview'][:50]!r} "
            f"tokens={sample['input_ids'].shape[0]} logits_shape={tuple(sample['top_k_logits'].shape)} "
            f"top_token_0={top_token_text!r} has_nan={has_nan} all_zero={all_zero}"
        )
        if has_nan or all_zero:
            raise RuntimeError("Corpus verification failed: invalid logits detected.")

    elapsed = time.perf_counter() - start
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    log(f"post_cleanup_current_rss_gb={current_rss_gb():.2f}")
    log(f"post_cleanup_peak_rss_gb={peak_rss_gb():.2f}")
    log("summary")
    log(f"total_examples={len(selected)}")
    log(f"corpus_size_gb={disk_usage_gb(output_dir):.2f}")
    log(f"generation_time_min={elapsed/60.0:.1f}")
    log(f"peak_memory_gb={peak_rss_gb():.2f}")
    log("corpus_ready_for_phase_2=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
