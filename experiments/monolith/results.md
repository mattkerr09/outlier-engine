# OUTLIER-ENGINE-MONOLITH-001: Results

## File Format

Single monolith binary with 4KB-aligned layout:
- **Header**: 4096 bytes — magic (`OEXS`), version, counts, per-projection sizes
- **Index**: 224 entries × 32 bytes = 7168 bytes (padded to 8192)
- **Data**: 224 expert blobs, each padded to 4096-byte boundary
- Layout: layer-interleaved (all layer-0 experts, then layer-1, etc.)

## Size Comparison

| Metric | Value |
|--------|-------|
| Individual files | 1,344 |
| Individual total | 8,702.4 MiB |
| Monolith size | 8,702.8 MiB |
| Overhead (index + padding) | 0.36 MiB (0.004%) |

Per-expert blob: 38.9 MiB (3 × 13.6 MB ternary + 3 × 2 B scale).
Padding per expert: 1,630 bytes to reach 4KB boundary.

## Benchmark Results

### Random Access (100 random experts)

| Method | Best Avg (ms/expert) | Speedup |
|--------|---------------------|---------|
| Individual files (6 opens) | 10.52 | 1.0x |
| **Monolith (1 seek+read)** | **6.12** | **1.72x** |

### Sequential Layer Access (layers 8-12)

| Method | Best Avg (ms/layer) | Speedup |
|--------|---------------------|---------|
| Individual files (48 opens/layer) | 40.0 | 1.0x |
| Monolith (1 bulk read/layer) | 55.1 | 0.73x |

## Analysis

**Random access: 1.72x speedup.** The monolith avoids 5 extra `open()`/`close()`
syscalls per expert load (6 files → 1 seek+read from an already-open fd). The
saved syscall overhead (~0.7 ms per file open) adds up: 6 opens × 0.7 ms ≈ 4.2 ms
saved, matching the observed 4.4 ms improvement.

**Sequential layer: 0.73x (slower).** The current `load_layer()` re-parses the
header and index for each call. More importantly, on macOS the unified buffer cache
handles 48 small reads from cached individual files efficiently — the 311 MiB bulk
read from the monolith doesn't benefit from the same cache locality. This could be
improved with a persistent file handle and mmap, but was kept simple for this prototype.

## Verification

- All 224 experts verified byte-for-byte against original files: **0 mismatches**
- All expert offsets confirmed 4096-byte aligned
- Layer interleaving confirmed: all layer-N experts contiguous, ordered by layer
- Tests: 11 new tests pass, 38 total (0 regressions)

## Recommendation: Worth Integrating?

**Conditional yes.** The monolith is worth integrating for specific use cases:

1. **Random expert access during inference**: 1.72x speedup with 0.004% size overhead
   is a clear win for the paging runtime's random access pattern.

2. **Reduced inode pressure**: 1,344 files → 1 file. On shared filesystems (NFS, FUSE)
   or containers, this eliminates metadata bottlenecks.

3. **Simpler deployment**: single file to distribute, no directory to manage.

**Not recommended** as a replacement for sequential layer loading unless the
implementation is extended with:
- Persistent file handle (avoid re-opening per call)
- Memory-mapped I/O for the data region
- Cached index table

**Next steps** if integrating:
- Add `MonolithExpertLoader` adapter that implements the same interface as the
  packed file loader in `paging.py`
- Add `repack --monolith` CLI command
- Benchmark on Linux (where `O_DIRECT` can bypass the buffer cache for fairer comparison)
