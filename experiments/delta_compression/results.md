# Delta Compression Results

## V1 Diagnosis

Checkpoint diagnosis was run on `Outlier-Ai/Outlier-10B` layer `8`.
The routed experts are duplicated in the checkpoint itself, not duplicated by the loader.
The safetensors file contains distinct expert keys and distinct byte ranges, but the payloads are bitwise identical.

## V2 Diverse Experts

Created `8` synthetic routed experts for layer `8` by applying dropout upcycling (`r=0.5`) to the shared FFN and quantizing each expert with scalar absmean ternarization.
- Average pairwise cosine similarity across synthesized experts: `0.499987`
- Min / max pairwise cosine similarity: `0.499883` / `0.500056`

## V2 Compression Findings

Reconstruction metrics below are measured in ternary-code space because the delta is defined on ternary shared/expert weights.
- Average delta sparsity (`expert_ternary - shared_ternary == 0`): `58.10%`
- Average differing positions vs shared ternary: `41.90%`
- Average reconstruction L2: `5367.4073`
- Average cosine similarity: `0.842443`
- Average dense delta TQ1_0 ratio: `1.000x`
- Average sparse ternary delta ratio (indices + packed delta values): `0.134x`
- Average sparse raw delta ratio (indices + 2-bit delta values): `0.132x`
- Average sparse transition-diff ratio (indices + 3-bit transition code): `0.127x`

## Claim 10b

No. Under this diverse-expert synthetic test, none of the measured delta encodings compressed better than the full TQ1_0 expert.

## Recommendation

This does not look worth pursuing as a broad patent non-provisional compression claim in its current form. The real bottleneck is position encoding: once experts are truly diverse, the sparse diff is too dense for index overhead to amortize.

## Per-Expert V2 Results

| Expert | Delta Sparsity | Recon L2 | Cosine | Full TQ1_0 MiB | Dense Delta Ratio | Sparse Ternary Ratio | Sparse Raw Ratio | Diff % | Transition Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 58.10% | 5367.3803 | 0.842444 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
| 1 | 58.11% | 5367.7396 | 0.842425 | 38.85 | 1.000x | 0.134x | 0.132x | 41.89% | 0.127x |
| 2 | 58.10% | 5367.4819 | 0.842444 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
| 3 | 58.10% | 5367.3211 | 0.842446 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
| 4 | 58.10% | 5367.4602 | 0.842452 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
| 5 | 58.09% | 5367.1352 | 0.842439 | 38.85 | 1.000x | 0.133x | 0.132x | 41.91% | 0.127x |
| 6 | 58.10% | 5367.4434 | 0.842452 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
| 7 | 58.10% | 5367.2965 | 0.842440 | 38.85 | 1.000x | 0.134x | 0.132x | 41.90% | 0.127x |
