#!/usr/bin/env python3
"""
OUTLIER-ENGINE-ENTROPY-CODING-001

Benchmark Zstd and LZ4 compression on TQ1_0-packed expert ternary files.

Measures:
  - Compression ratio at each level
  - Compression + decompression throughput (GB/s)
  - Shannon entropy of raw bytes
  - Roundtrip correctness

Usage:
    cd ~/outlier-engine && source .venv/bin/activate
    python experiments/entropy_coding/compress_experts.py 2>&1 | tee experiments/entropy_coding/results.txt
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# ── stdlib compression ──────────────────────────────────────────────────────
import zlib

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("WARNING: zstandard not installed — install with: pip install zstandard", file=sys.stderr)

try:
    import lz4.frame as lz4_frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("WARNING: lz4 not installed — install with: pip install lz4", file=sys.stderr)

# ── config ───────────────────────────────────────────────────────────────────
PACKED_DIR = Path(__file__).resolve().parents[2] / "packed_experts"
N_SAMPLE = 12       # number of expert ternary files to benchmark (spread across layers)
N_DECOMP_REPS = 5   # decompression reps for timing stability
MIN_DECOMP_GBPS = 3.0  # target threshold for "fast enough" (GB/s)

ZSTD_LEVELS = [1, 3, 6, 9]
LZ4_LEVELS  = [0, 4, 9]   # 0=default, 4=fast, 9=high


# ── helpers ──────────────────────────────────────────────────────────────────

def shannon_entropy_bits_per_byte(data: bytes) -> float:
    """Shannon entropy H in bits per byte (max = 8)."""
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def bits_per_weight_tq10(original_bytes: int, compressed_bytes: int) -> float:
    """Convert compressed bytes → bits per ternary weight.

    TQ1_0 packs 5 ternary values per byte (base-3, stored as uint8).
    So original_bytes represents: original_bytes * 5 ternary values.
    Compressed bits per weight = compressed_bytes * 8 / (original_bytes * 5).
    """
    n_weights = original_bytes * 5
    return (compressed_bytes * 8) / n_weights


def benchmark_zstd(data: bytes, level: int, n_reps: int) -> dict:
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()

    # Warmup
    compressed = cctx.compress(data)

    # Compression timing
    t0 = time.perf_counter()
    compressed = cctx.compress(data)
    comp_s = time.perf_counter() - t0

    # Decompression timing (N_DECOMP_REPS for stability)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        decompressed = dctx.decompress(compressed)
    decomp_s = (time.perf_counter() - t0) / n_reps

    assert decompressed == data, "zstd roundtrip FAILED"

    orig_mb  = len(data) / 1e6
    comp_mb  = len(compressed) / 1e6
    ratio    = len(data) / len(compressed)
    comp_gbps   = (orig_mb / 1e3) / max(comp_s, 1e-9)
    decomp_gbps = (orig_mb / 1e3) / max(decomp_s, 1e-9)
    bpw         = bits_per_weight_tq10(len(data), len(compressed))

    return {
        "codec":        f"zstd-{level}",
        "orig_mb":      orig_mb,
        "comp_mb":      comp_mb,
        "ratio":        ratio,
        "bpw":          bpw,
        "comp_gbps":    comp_gbps,
        "decomp_gbps":  decomp_gbps,
        "roundtrip_ok": True,
    }


def benchmark_lz4(data: bytes, compression_level: int, n_reps: int) -> dict:
    label = {0: "lz4-default", 4: "lz4-fast4", 9: "lz4-hc9"}.get(compression_level, f"lz4-{compression_level}")

    # Warmup
    compressed = lz4_frame.compress(data, compression_level=compression_level)

    # Compression
    t0 = time.perf_counter()
    compressed = lz4_frame.compress(data, compression_level=compression_level)
    comp_s = time.perf_counter() - t0

    # Decompression
    t0 = time.perf_counter()
    for _ in range(n_reps):
        decompressed = lz4_frame.decompress(compressed)
    decomp_s = (time.perf_counter() - t0) / n_reps

    assert decompressed == data, f"lz4 roundtrip FAILED (level={compression_level})"

    orig_mb  = len(data) / 1e6
    comp_mb  = len(compressed) / 1e6
    ratio    = len(data) / len(compressed)
    comp_gbps   = (orig_mb / 1e3) / max(comp_s, 1e-9)
    decomp_gbps = (orig_mb / 1e3) / max(decomp_s, 1e-9)
    bpw         = bits_per_weight_tq10(len(data), len(compressed))

    return {
        "codec":        label,
        "orig_mb":      orig_mb,
        "comp_mb":      comp_mb,
        "ratio":        ratio,
        "bpw":          bpw,
        "comp_gbps":    comp_gbps,
        "decomp_gbps":  decomp_gbps,
        "roundtrip_ok": True,
    }


def benchmark_zlib(data: bytes, level: int, n_reps: int) -> dict:
    t0 = time.perf_counter()
    compressed = zlib.compress(data, level=level)
    comp_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_reps):
        decompressed = zlib.decompress(compressed)
    decomp_s = (time.perf_counter() - t0) / n_reps

    assert decompressed == data, f"zlib roundtrip FAILED (level={level})"

    orig_mb  = len(data) / 1e6
    comp_mb  = len(compressed) / 1e6
    ratio    = len(data) / len(compressed)
    comp_gbps   = (orig_mb / 1e3) / max(comp_s, 1e-9)
    decomp_gbps = (orig_mb / 1e3) / max(decomp_s, 1e-9)
    bpw         = bits_per_weight_tq10(len(data), len(compressed))

    return {
        "codec":        f"zlib-{level}",
        "orig_mb":      orig_mb,
        "comp_mb":      comp_mb,
        "ratio":        ratio,
        "bpw":          bpw,
        "comp_gbps":    comp_gbps,
        "decomp_gbps":  decomp_gbps,
        "roundtrip_ok": True,
    }


def fmt_row(r: dict) -> str:
    ok = "✓" if r["roundtrip_ok"] else "✗"
    fast = "✓" if r["decomp_gbps"] >= MIN_DECOMP_GBPS else " "
    return (
        f"  {r['codec']:<16} {r['orig_mb']:6.1f} MB → {r['comp_mb']:6.1f} MB "
        f"ratio={r['ratio']:.3f}x  bpw={r['bpw']:.3f}  "
        f"comp={r['comp_gbps']:.2f} GB/s  decomp={r['decomp_gbps']:.2f} GB/s  "
        f"rt={ok}  fast={fast}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not PACKED_DIR.exists():
        sys.exit(f"ERROR: packed_experts dir not found at {PACKED_DIR}")

    all_ternary = sorted(PACKED_DIR.glob("*_ternary.bin"))
    if not all_ternary:
        sys.exit(f"ERROR: no *_ternary.bin files found in {PACKED_DIR}")

    print(f"found {len(all_ternary)} ternary expert files in {PACKED_DIR}")

    # Sample N_SAMPLE files spread evenly across the list
    step = max(1, len(all_ternary) // N_SAMPLE)
    sample_files = all_ternary[::step][:N_SAMPLE]
    print(f"sampling {len(sample_files)} files (every {step}th)")
    print()

    # Accumulators: codec → list of result dicts
    agg: dict[str, list[dict]] = {}

    entropy_values: list[float] = []
    total_ternary_gb = len(all_ternary) * (all_ternary[0].stat().st_size if all_ternary else 0) / 1e9

    for i, fpath in enumerate(sample_files):
        data = fpath.read_bytes()
        size_mb = len(data) / 1e6
        h = shannon_entropy_bits_per_byte(data)
        entropy_values.append(h)

        print(f"[{i+1}/{len(sample_files)}] {fpath.name}  ({size_mb:.1f} MB, entropy={h:.4f} bits/byte)")

        # TQ1_0 theoretical: log2(3) = 1.585 bits per weight
        # packed at 5 values/byte → 1.585/5*8 = 2.536 bits/byte entropy ceiling
        # but actual stored uses base-3 arithmetic → real entropy should be < 8
        bpw_raw = bits_per_weight_tq10(len(data), len(data))
        print(f"  raw_bpw={bpw_raw:.3f}  (theoretical TQ1_0 ceiling: 1.585 bpw)")

        results = []

        # ── Zstd ──
        if HAS_ZSTD:
            for lvl in ZSTD_LEVELS:
                r = benchmark_zstd(data, lvl, N_DECOMP_REPS)
                results.append(r)
                agg.setdefault(r["codec"], []).append(r)
                print(fmt_row(r))

        # ── LZ4 ──
        if HAS_LZ4:
            for lvl in LZ4_LEVELS:
                r = benchmark_lz4(data, lvl, N_DECOMP_REPS)
                results.append(r)
                agg.setdefault(r["codec"], []).append(r)
                print(fmt_row(r))

        # ── zlib-6 for reference ──
        r = benchmark_zlib(data, 6, N_DECOMP_REPS)
        results.append(r)
        agg.setdefault(r["codec"], []).append(r)
        print(fmt_row(r))

        print()

    # ── Aggregate averages ──
    print("=" * 90)
    print("AVERAGES ACROSS ALL SAMPLED EXPERTS")
    print("=" * 90)
    print(f"  {'codec':<16} {'ratio':>8}  {'bpw':>6}  {'comp GB/s':>10}  {'decomp GB/s':>12}  {'fast?':>6}")
    print(f"  {'-'*16} {'-'*8}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*6}")

    avg_results = []
    for codec, rlist in sorted(agg.items()):
        avg_ratio  = sum(r["ratio"]       for r in rlist) / len(rlist)
        avg_bpw    = sum(r["bpw"]         for r in rlist) / len(rlist)
        avg_comp   = sum(r["comp_gbps"]   for r in rlist) / len(rlist)
        avg_decomp = sum(r["decomp_gbps"] for r in rlist) / len(rlist)
        fast = avg_decomp >= MIN_DECOMP_GBPS
        avg_results.append({
            "codec": codec, "ratio": avg_ratio, "bpw": avg_bpw,
            "comp_gbps": avg_comp, "decomp_gbps": avg_decomp, "fast": fast,
        })
        flag = "✓ FAST" if fast else "  slow"
        print(f"  {codec:<16} {avg_ratio:>8.3f}x {avg_bpw:>6.3f}  {avg_comp:>10.2f}  {avg_decomp:>12.2f}  {flag}")

    print()

    # ── Entropy summary ──
    avg_entropy = sum(entropy_values) / max(len(entropy_values), 1)
    tq10_theoretical_bpw = math.log2(3)      # 1.585 bits/ternary weight
    entropy_bpw = avg_entropy / 5            # bits/weight from byte entropy (/5 vals/byte)
    print(f"Shannon entropy (byte-level): avg={avg_entropy:.4f} bits/byte  (max=8)")
    print(f"  → bits per ternary weight (entropy floor): {entropy_bpw:.4f}")
    print(f"  → TQ1_0 theoretical ceiling: {tq10_theoretical_bpw:.4f} bpw")
    print(f"  → TQ1_0 raw (uncompressed): 1.6 bpw  (5 vals × 8 bits / 25 slots per 5 bytes)")
    print()

    # ── Recommendation ──
    print("=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    fast_options = [r for r in avg_results if r["fast"]]
    if fast_options:
        best = min(fast_options, key=lambda r: r["bpw"])
        print(f"  Best codec meeting >{MIN_DECOMP_GBPS} GB/s decomp threshold: {best['codec']}")
        print(f"    ratio={best['ratio']:.3f}x  bpw={best['bpw']:.3f}  decomp={best['decomp_gbps']:.2f} GB/s")
    else:
        best = min(avg_results, key=lambda r: r["bpw"])
        print(f"  No codec meets >{MIN_DECOMP_GBPS} GB/s threshold. Best ratio: {best['codec']}")

    # ── Storage impact ──
    print()
    print("STORAGE IMPACT (Outlier-10B, all ternary experts)")
    n_experts_total = len(all_ternary)
    expert_bytes    = all_ternary[0].stat().st_size if all_ternary else 0
    raw_gb          = n_experts_total * expert_bytes / 1e9
    print(f"  Ternary expert files: {n_experts_total}  ×  {expert_bytes/1e6:.1f} MB  =  {raw_gb:.1f} GB")

    for r in avg_results:
        saved_gb = raw_gb * (1 - 1/r["ratio"])
        comp_gb  = raw_gb / r["ratio"]
        print(f"  {r['codec']:<16}  {raw_gb:.1f} GB → {comp_gb:.1f} GB  (save {saved_gb:.1f} GB, {(1-1/r['ratio'])*100:.1f}%)")

    print()
    print(f"  NVMe threshold: >{MIN_DECOMP_GBPS} GB/s decompression needed to not bottleneck 5 GB/s SSD")


if __name__ == "__main__":
    main()
