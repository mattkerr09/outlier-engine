# Sherry 3:4 on Ternary Experts

## Setup

This prototype evaluates Sherry-style 3:4 structured sparsity on the existing layer-8 diverse-expert setup from [`experiments/delta_compression/results.md`](/Users/matthewkerr/outlier-engine/experiments/delta_compression/results.md).

Why that assumption:
- The checkpointed routed experts in `Outlier-Ai/Outlier-10B` are duplicated ternary tensors.
- They store `*_ternary` and scalar `*_scale`, but not the pre-rounding float magnitudes needed for the exact "zero the smallest original magnitude" rule.
- The synthetic layer-8 experts already used in the delta-compression experiment are deterministic and derived from the real shared FP16 weights, so they let us apply the Sherry rule exactly.

## Single Expert

Sampled expert: layer `8`, expert `0`.

- Natural sparsity before enforcement: `58.10%`
- Sparsity after 3:4 enforcement: `58.87%`
- Groups needing enforcement: `3.09%`
- Newly forced-zero weights: `0.773%`
- Cosine similarity vs original ternary expert: `0.990732`
- L2 error: `16.3546`

Interpretation:
- The sparsification step itself is gentle.
- Only about `0.77%` of weights are newly zeroed, and cosine stays comfortably above the `>0.95` target.

## Layer 8 Aggregate

Across all `8` synthetic experts in layer `8`:

- Average natural sparsity: `58.10%`
- Average enforced sparsity: `58.87%`
- Average groups needing enforcement: `3.09%`
- Average forced-zero weights: `0.773%`
- Average cosine similarity: `0.990736`
- Average L2 error: `16.3542`

Post-enforcement group structure is the key result:

- Exactly `1` zero: `20.19%` of groups
- Exactly `2` zeros: `35.54%` of groups
- Exactly `3` zeros: `32.85%` of groups
- Exactly `4` zeros: `11.42%` of groups

That means `79.81%` of groups still have `2+` zeros after 3:4 enforcement.

## Packing Results

Implemented in [`sherry_packing.py`](/Users/matthewkerr/outlier-engine/experiments/sherry/sherry_packing.py):

- A strict 5-bit Sherry codec for tensors with exactly one zero per 4-weight group
- A fallback-capable hybrid codec that roundtrips the real enforced tensors exactly

Roundtrip checks:

- Strict codec roundtrip: `pass`
- Hybrid codec roundtrip on enforced sample tensor: `pass`

Compression against current TQ1_0:

| Metric | TQ1_0 | Strict Sherry Ideal | Exact Hybrid on Real Enforced Experts |
| --- | --- | --- | --- |
| Bits / weight | `1.6000` | `1.2500` | `1.5603` |
| Size ratio vs TQ1_0 | `1.000x` | `0.7813x` | `0.9752x` |
| Size reduction vs TQ1_0 | `0.00%` | `21.88%` | `2.48%` |

Layer-8 totals across all `8` experts:

- TQ1_0: `310.80 MiB`
- Strict Sherry ideal: `242.81 MiB`
- Exact hybrid on real enforced experts: `303.09 MiB`

## Storage Estimate

Estimated routed-expert storage across all `28` layers:

- Current TQ1_0 expert store: `8.50 GiB`
- Strict Sherry ideal: `6.64 GiB`
- Exact hybrid on real enforced experts: `8.29 GiB`

Estimated routed-expert savings:

- Ideal exact-one-zero Sherry: `1.86 GiB`
- Actual exact hybrid on current zero-heavy ternary experts: `215.93 MiB`

Projected to an Outlier-150B model by simple 15x scaling:

- Ideal exact-one-zero Sherry savings: about `27.89 GiB`
- Actual exact hybrid savings at the same group distribution: about `3.16 GiB`

Important caveat:
- These are routed-expert storage savings only.
- Shared expert, attention, embeddings, and router weights do not change here, so whole-model savings would be smaller than the routed-expert numbers above.

## Answer

### Does Sherry 3:4 achieve the expected 23% reduction?

Not on these existing ternary experts.

- If the tensor truly fit the strict Sherry assumption of exactly one zero per group, yes: the math gives `1.25` bits/weight and `21.88%` smaller than TQ1_0.
- On the actual Outlier-style ternary experts we tested, most groups already contain `2+` zeros, so the strict 5-bit scheme is not directly applicable.
- An exact roundtrippable hybrid format only reaches about `2.48%` reduction.

### What's the quality cost?

Low.

- Average cosine similarity is `0.990736`.
- Only `3.09%` of groups need intervention.
- Only `0.773%` of weights are newly zeroed.

### Worth integrating into the engine?

Not in the current form.

Recommendation:
- Do **not** integrate the strict 5-bit Sherry packer into the engine for the current ternary expert distribution.
- The sparsification rule itself looks safe enough to keep exploring.
- The blocker is packing compatibility: current absmean-ternary experts are too zero-heavy, so they do not land in the "exactly one zero + three signs" regime that makes Sherry attractive.
- This becomes interesting again only if training or reparameterization changes the post-quantization group distribution toward exactly one zero per 4-weight block.
