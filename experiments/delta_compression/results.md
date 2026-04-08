# Delta Compression Results

## Summary

Sampled `24` routed experts from layers `0, 8, 16` of `Outlier-Ai/Outlier-10B` from local snapshot `/Users/matthewkerr/.cache/huggingface/hub/models--Outlier-Ai--Outlier-10B/snapshots/11ff21cad8faa97dafab2362f20bf790da7f3ae3`.
The experiment tested the hypothesis `expert ~= shared_ffn + ternary_delta`, where the delta was ternarized with scalar absmean quantization.

- Runtime: `30.8s`
- Average sparsity: `38.85%`
- Average reconstruction error (L2): `87.1834`
- Average cosine similarity: `0.911030`
- Average compression ratio, full expert TQ1_0 / delta TQ1_0: `1.000x`
- Experts above 80% sparsity: `0`

## Checkpoint Notes

Exact tensor comparison across gate/up/down ternary payloads and scales found:
- layer 0: 1 unique routed expert payloads across 8 experts
- layer 8: 1 unique routed expert payloads across 8 experts
- layer 16: 1 unique routed expert payloads across 8 experts

## Recommendation

Not worth pursuing in its current storage format. Plain TQ1_0 delta packing does not beat full-expert TQ1_0, and the sparse RLE upside is too small to justify a patent non-provisional claim yet.

## Per-Expert Results

| Layer | Expert | Sparsity | Recon L2 | Cosine | Full TQ1_0 MiB | Delta TQ1_0 MiB | Ratio | RLE TQ1_0 MiB | RLE Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 1 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 2 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 3 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 4 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 5 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 6 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 0 | 7 | 45.32% | 92.3690 | 0.877697 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 0 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 1 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 2 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 3 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 4 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 5 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 6 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 8 | 7 | 35.46% | 83.8699 | 0.929094 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 0 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 1 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 2 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 3 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 4 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 5 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 6 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
| 16 | 7 | 35.76% | 85.3113 | 0.926298 | 38.85 | 38.85 | 1.000x | n/a | n/a |
