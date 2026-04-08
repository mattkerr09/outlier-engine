"""
Benchmark: individual files vs monolith expert store.

Tests random access and sequential layer access patterns.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from outlier_engine.expert_store import (
    ExpertStore,
    SUB_FILE_ORDER,
    HEADER_SIZE,
    ALIGNMENT,
    _align,
    _key_for,
)

EXPERT_DIR = Path.home() / "outlier-engine" / "packed_experts"
MONOLITH_PATH = Path(__file__).resolve().parent / "experts.bin"

NUM_LAYERS = 28
NUM_EXPERTS = 8
N_RANDOM_SAMPLES = 100
LAYER_RANGE = range(8, 13)  # layers 8-12 for sequential test


def bench_random_individual(pairs, index):
    """Random access via individual files."""
    times = []
    for layer, expert in pairs:
        t0 = time.perf_counter()
        blob = bytearray()
        for sub in SUB_FILE_ORDER:
            key = _key_for(layer, expert, sub)
            info = index[key]
            path = EXPERT_DIR / info["file"]
            with open(path, "rb") as f:
                blob.extend(f.read())
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_random_monolith(pairs, store_path):
    """Random access via monolith (open once, seek per expert)."""
    # Pre-read header and index once
    with open(store_path, "rb") as f:
        hdr = ExpertStore._read_header(f)
        idx = ExpertStore._read_index(f, hdr["num_entries"])

        times = []
        for layer, expert in pairs:
            offset, size = idx[(layer, expert)]
            t0 = time.perf_counter()
            f.seek(offset)
            _ = f.read(size)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return times


def bench_sequential_individual(layers, index):
    """Sequential layer access via individual files."""
    times = []
    for layer in layers:
        t0 = time.perf_counter()
        for expert in range(NUM_EXPERTS):
            for sub in SUB_FILE_ORDER:
                key = _key_for(layer, expert, sub)
                info = index[key]
                path = EXPERT_DIR / info["file"]
                with open(path, "rb") as f:
                    _ = f.read()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def bench_sequential_monolith(layers, store_path):
    """Sequential layer access via monolith (one read per layer)."""
    times = []
    for layer in layers:
        t0 = time.perf_counter()
        _ = ExpertStore.load_layer(store_path, layer)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main():
    print("=" * 70)
    print("MONOLITH BENCHMARK: Individual Files vs Single-File Expert Store")
    print("=" * 70)

    # Check prerequisites
    if not EXPERT_DIR.exists():
        print(f"ERROR: Expert directory not found: {EXPERT_DIR}")
        return
    if not MONOLITH_PATH.exists():
        print(f"Packing experts into {MONOLITH_PATH}...")
        stats = ExpertStore.pack(str(EXPERT_DIR), str(MONOLITH_PATH))
        print(f"  Packed {stats['num_entries']} experts, total {stats['total_size'] / 1024 / 1024:.1f} MiB")
    else:
        print(f"Monolith already exists: {MONOLITH_PATH}")

    index = json.loads((EXPERT_DIR / "index.json").read_text(encoding="utf-8"))

    # File counts and sizes
    individual_files = list(EXPERT_DIR.glob("*.bin"))
    individual_size = sum(f.stat().st_size for f in individual_files)
    monolith_size = MONOLITH_PATH.stat().st_size
    print(f"\n  Individual files: {len(individual_files)}")
    print(f"  Individual total size: {individual_size / 1024 / 1024:.1f} MiB")
    print(f"  Monolith size: {monolith_size / 1024 / 1024:.1f} MiB")
    print(f"  Overhead: {(monolith_size - individual_size) / 1024 / 1024:.2f} MiB "
          f"({(monolith_size - individual_size) / individual_size * 100:.2f}%)")

    # Drop OS file cache (best-effort)
    print("\n  Note: file cache effects may influence results.")
    print("  Running warmup pass first...\n")

    # Warmup — touch monolith and a few individual files
    with open(MONOLITH_PATH, "rb") as f:
        f.read(4096)
    for sub in SUB_FILE_ORDER:
        path = EXPERT_DIR / index[_key_for(0, 0, sub)]["file"]
        with open(path, "rb") as f:
            f.read()

    # Generate random access pairs
    random.seed(42)
    all_pairs = [(l, e) for l in range(NUM_LAYERS) for e in range(NUM_EXPERTS)]
    random_pairs = random.sample(all_pairs, N_RANDOM_SAMPLES)

    # --- Random access benchmark ---
    print("-" * 70)
    print(f"RANDOM ACCESS ({N_RANDOM_SAMPLES} samples)")
    print("-" * 70)

    # Run 3 iterations and take the best
    best_individual = None
    best_monolith = None

    for iteration in range(3):
        random.shuffle(random_pairs)

        t_ind = bench_random_individual(random_pairs, index)
        t_mon = bench_random_monolith(random_pairs, MONOLITH_PATH)

        avg_ind = sum(t_ind) / len(t_ind)
        avg_mon = sum(t_mon) / len(t_mon)

        if best_individual is None or avg_ind < best_individual:
            best_individual = avg_ind
        if best_monolith is None or avg_mon < best_monolith:
            best_monolith = avg_mon

        print(f"  Iter {iteration + 1}: individual={avg_ind * 1000:.3f}ms, "
              f"monolith={avg_mon * 1000:.3f}ms, "
              f"speedup={avg_ind / max(avg_mon, 1e-9):.2f}x")

    print(f"\n  Best individual avg: {best_individual * 1000:.3f} ms/expert")
    print(f"  Best monolith avg:   {best_monolith * 1000:.3f} ms/expert")
    speedup_random = best_individual / max(best_monolith, 1e-9)
    print(f"  Speedup: {speedup_random:.2f}x")

    # --- Sequential layer access benchmark ---
    print(f"\n" + "-" * 70)
    layers_list = list(LAYER_RANGE)
    n_layer_files = len(layers_list) * NUM_EXPERTS * len(SUB_FILE_ORDER)
    print(f"SEQUENTIAL LAYER ACCESS (layers {layers_list[0]}-{layers_list[-1]}, "
          f"{n_layer_files} file opens vs {len(layers_list)} reads)")
    print("-" * 70)

    best_seq_ind = None
    best_seq_mon = None

    for iteration in range(3):
        t_seq_ind = bench_sequential_individual(layers_list, index)
        t_seq_mon = bench_sequential_monolith(layers_list, MONOLITH_PATH)

        avg_seq_ind = sum(t_seq_ind) / len(t_seq_ind)
        avg_seq_mon = sum(t_seq_mon) / len(t_seq_mon)

        if best_seq_ind is None or avg_seq_ind < best_seq_ind:
            best_seq_ind = avg_seq_ind
        if best_seq_mon is None or avg_seq_mon < best_seq_mon:
            best_seq_mon = avg_seq_mon

        print(f"  Iter {iteration + 1}: individual={avg_seq_ind * 1000:.1f}ms/layer, "
              f"monolith={avg_seq_mon * 1000:.1f}ms/layer, "
              f"speedup={avg_seq_ind / max(avg_seq_mon, 1e-9):.2f}x")

    print(f"\n  Best individual avg: {best_seq_ind * 1000:.1f} ms/layer")
    print(f"  Best monolith avg:   {best_seq_mon * 1000:.1f} ms/layer")
    speedup_seq = best_seq_ind / max(best_seq_mon, 1e-9)
    print(f"  Speedup: {speedup_seq:.2f}x")

    # --- Summary ---
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Random access speedup:     {speedup_random:.2f}x")
    print(f"  Sequential layer speedup:  {speedup_seq:.2f}x")
    print(f"  Individual files:          {len(individual_files)}")
    print(f"  Individual total:          {individual_size / 1024 / 1024:.1f} MiB")
    print(f"  Monolith size:             {monolith_size / 1024 / 1024:.1f} MiB")
    print(f"  Overhead:                  {(monolith_size - individual_size) / 1024 / 1024:.2f} MiB "
          f"({(monolith_size - individual_size) / individual_size * 100:.3f}%)")


if __name__ == "__main__":
    main()
