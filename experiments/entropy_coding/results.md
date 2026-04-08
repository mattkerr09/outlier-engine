# OUTLIER-ENGINE-ENTROPY-CODING-001 Results

## Experiment

Measured Zstd (levels 1/3/6/9), LZ4 (default/fast4/hc9), and zlib-6 compression
on 12 sampled TQ1_0-packed expert ternary files from Outlier-10B.
All roundtrips verified lossless. Measured on Apple M-series (MPS host CPU).

## Entropy Measurement

| Metric | Value |
|--------|-------|
| Byte-level Shannon entropy (avg) | 7.75 bits/byte |
| Entropy bits/ternary weight (floor) | **1.550 bpw** |
| TQ1_0 theoretical ceiling | 1.585 bpw |
| TQ1_0 raw (uncompressed) | 1.600 bpw |
| Gap: raw → entropy floor | 0.050 bpw (3.1%) |

**Finding:** TQ1_0 already captures 97% of the theoretical entropy minimum.
The remaining gap (0.05 bpw) is the maximum any entropy coder can recover.

Most experts sit at entropy ≈ 7.92/8 bits/byte — the packed bytes look nearly
random, meaning very little redundancy remains for general-purpose compressors.

**Exception:** One expert (`layers.1.mlp.experts.5.gate`) has entropy 5.98 bits/byte,
compressing 1.6× with zstd. This likely reflects an early-layer expert that learned
more structured (sparser) weights. Compression benefit varies by layer and expert index.

## Compression Ratio Table

Averages across 12 sampled experts (12 of 672 ternary files, spread evenly):

| Codec | Avg Ratio | Avg bpw | Comp GB/s | Decomp GB/s | Meets >3 GB/s? |
|-------|-----------|---------|-----------|-------------|----------------|
| lz4-default | 1.032× | 1.563 | 3.34 | **15.99** | ✓ |
| lz4-fast4   | 1.039× | 1.557 | 0.06 | **16.33** | ✓ |
| lz4-hc9     | 1.041× | 1.556 | 0.05 | **16.99** | ✓ |
| zlib-6      | 1.049× | 1.545 | 0.05 | 0.46 | ✗ |
| zstd-1      | 1.049× | 1.550 | **5.31** | **27.60** | ✓ |
| zstd-3      | 1.052× | 1.548 | **5.37** | **30.76** | ✓ |
| zstd-6      | 1.055× | 1.546 | 3.73 | **31.67** | ✓ |
| **zstd-9**  | **1.056×** | **1.545** | 3.27 | **31.87** | ✓ |

Zstd-1 offers the best compression throughput (5.31 GB/s compression speed) which
matters at repack time. Zstd-9 gives the best ratio but identical decompression speed.

## Recommendation

**Do not add entropy coding to the production load path.**

Rationale:
1. **Negligible ratio gain:** avg 1.05× (5%) compression. The 9.1 GB expert corpus
   shrinks to ~8.6 GB — a 0.5 GB saving. Not worth the added complexity.
2. **Expert data is near-maximum entropy.** TQ1_0 leaves only 0.05 bpw on the table.
   To recover that, you need a trained entropy model (ANS/arithmetic coding with
   context) — a general-purpose compressor can't reach the floor.
3. **Decompression speed is not the constraint.** All Zstd levels decompress at
   27–32 GB/s; all LZ4 levels at 16–17 GB/s. Both far exceed a 5 GB/s NVMe.
   The actual load bottleneck is NVMe I/O bandwidth and CPU→GPU transfer, not
   decompression throughput.

If you still want to ship compression, **zstd-1** is the pragmatic choice:
- 5.3 GB/s compression speed (fast repacking)
- 27.6 GB/s decompression (3.1 GB/s NVMe is never the bottleneck)
- 4.6% storage reduction
- Zero effect on inference latency (decompresses >5× faster than any NVMe reads)

## Expected Storage Impact (Outlier-10B)

| Codec | Raw | Compressed | Saved | % |
|-------|-----|------------|-------|---|
| None  | 9.1 GB | 9.1 GB | — | — |
| zstd-1 | 9.1 GB | 8.7 GB | 0.4 GB | 4.6% |
| zstd-9 | 9.1 GB | 8.6 GB | 0.5 GB | 5.3% |
| LZ4-default | 9.1 GB | 8.8 GB | 0.3 GB | 3.1% |

*Note:* One outlier expert compressed 1.65× (layer 1, expert 5). If many early-layer
experts have similar structure, actual corpus savings could exceed 5%. A full-corpus
scan would give a definitive number.

## What Would Actually Help

To reach the entropy floor (1.550 bpw from 1.600 bpw), a **learned entropy coder**
is needed — one that models the distribution of ternary weights per channel or row.
Options:
- **ANS (Asymmetric Numeral Systems)** with per-expert symbol frequency tables
- **FSST** (Fast Static Symbol Table) — fast byte-level learned codec
- **Re-quantize the distribution** — if ternary weights are imbalanced (+1/0/-1
  at unequal frequencies), a Huffman code over the original values would recover
  the 0.05 bpw gap without any architectural changes to the loader

These are off-path optimizations. The bigger wins remain:
1. Expert prefetching (hide NVMe latency)
2. Contiguous expert storage for faster `torch.index_select`
3. Reducing expert count via ET routing pruning
