# TMAC Paged Inference Experiments

## Goal

Close the remaining gap between paged decode on Apple MPS and the non-paged Hugging Face path.

Current reference points:

- Non-paged decode: about `59 ms/token`
- Best paged decode before this sprint: `0.80 s/token` for warm tokens 3-5

## Layer 1: Match Hugging Face's Fast Path

I first profiled a single warm-cache expert invocation to check whether the expert math itself was the bottleneck.

From [`profile_output.log`](./profile_output.log):

- hidden dtype: `torch.float16`
- hidden device: `mps:0`
- weight dtype: `torch.float16`
- weight device: `mps:0`
- lookup avg: `0.021 ms`
- gate matmul avg: `0.028 ms`
- up matmul avg: `0.022 ms`
- activation avg: `0.075 ms`
- down matmul avg: `0.038 ms`
- combine avg: `0.244 ms`
- warm expert total avg: `0.429 ms`

Conclusion:

- The hot expert path is already on MPS.
- The hot expert path is already using torch ops.
- The hot expert path is not spending meaningful time in a single expert matmul.

I also tried more aggressive runtime changes during this layer, including fully materialized shared expert float weights. That regressed badly.

From [`bench_layer1.log`](./bench_layer1.log):

- token 1: `50.84 s`
- token 2-5 avg: `13.08 s`
- token 3-5 avg: `9.74 s`

So the Layer 1 experiments did not improve the real decode path.

## Layer 2: Ternary-Aware Matmul

I implemented and benchmarked three approaches in [`bench_ternary_matmul.py`](./bench_ternary_matmul.py):

1. Standard dense float16 linear on MPS
2. Ternary mask linear using positive/negative masks
3. Packed mask linear using bit-packed boolean masks

From [`bench_matmul_output.log`](./bench_matmul_output.log):

- standard float linear: `0.181 ms`
- ternary mask linear: `0.312 ms`
- packed mask linear: `18.707 ms`

Conclusion:

- On Apple MPS, plain dense float16 linear is faster than pure PyTorch ternary-aware matmul.
- Packed mask decoding is dramatically slower in Python/PyTorch.
- The fastest runtime path remains the existing float16 hot-cache path.

Because of that, I did **not** integrate the ternary mask path into the main paged runtime.

[`bench_layer2.log`](./bench_layer2.log) records that decision and preserves the best-known benchmark.

## Layer 3: LUT / T-MAC Style Kernel

I did not implement the full LUT kernel in Python.

Reason:

- Layer 2 already showed that pure PyTorch ternary-specialized kernels lose to MPS dense float16 linear.
- A Python LUT implementation would add even more interpreter overhead and would not be a meaningful proxy for a native Metal kernel.

## Final Result

The fastest path after this sprint is still the existing paged MPS runtime, preserved in [`paged_bench_results_v5.log`](../../paged_bench_results_v5.log):

- token 1: `48.29 s`
- token 2: `10.46 s`
- token 3: `1.07 s`
- token 4: `0.68 s`
- token 5: `0.66 s`
- token 3-5 avg: `0.80 s`
- token 2-5 avg: `3.22 s`

## Remaining Gap

Compared with the non-paged baseline:

- non-paged: `0.059 s/token`
- best paged warm path: `0.80 s/token`
- remaining gap: about `13.6x`

## What a Native Metal Kernel Needs to Do

The remaining work is not "replace float matmul with more Python."

To close the rest of the gap, a native Metal kernel would need to:

1. Consume packed ternary weights directly, without unpacking into standard dense tensors in Python.
2. Fuse dequantization with accumulation.
3. Batch multiple routed experts together in one command buffer.
4. Avoid repeated Python dispatch around per-layer expert selection.
5. Reuse hot expert state on-device across tokens.

In short:

- Pure PyTorch ternary math on MPS is not enough.
- A real Metal kernel is the next credible path to close the remaining gap.
