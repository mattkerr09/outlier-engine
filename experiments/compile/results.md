# Compile Experiments on Paged MPS Inference

## Goal

Test whether `torch.compile` or related compilation strategies can remove enough Python overhead from paged inference to move warm-token latency from `0.80 s` toward `0.10 s`.

Reference points:

- Non-paged decode: about `59 ms/token`
- Best paged warm decode before this sprint: `0.80 s/token` for tokens 3-5

## What We Measured

### 1. Isolated MLP compile sanity check on MPS

From [`compile_test_output.log`](./compile_test_output.log):

- eager: `0.702 ms`
- `torch.compile`: `0.716 ms`
- `torch.jit.trace`: `0.652 ms`

Conclusion:

- `torch.compile` works on MPS in this environment.
- On a simple dense MLP, compile is basically flat.
- `torch.jit.trace` is slightly faster, but the gain is small.

### 2. Hot expert math only

From [`compile_paged_output.log`](./compile_paged_output.log):

- expert math eager: `0.759 ms`
- expert math compile: `0.747 ms`
- fused module eager: `0.784 ms`
- fused module compile: `0.689 ms`
- fused module JIT trace: `0.682 ms`

Conclusion:

- The narrow expert math block is already fast.
- Compile and JIT only improve it by about `0.07-0.10 ms`.
- That is nowhere near enough to explain or eliminate the full `0.80 s/token` warm decode cost.

### 3. Captured paged MLP on a real warm hidden state

Using the actual monkey-patched paged MLP from the last transformer layer, on a captured warm hidden state:

- paged MLP eager: `26.933 ms`
- paged MLP compile: `12.539 ms`

This is the biggest win from the sprint.

Conclusion:

- `torch.compile` materially reduces the cost of the Python-heavy paged MLP itself on a fixed input.
- The fixed-input MLP call got about `2.15x` faster.

## What Failed

I also tried the broader paged-path experiment earlier in this sprint, and it failed before a useful token benchmark:

- Full paged warmup + compile attempt hit MPS OOM when using the larger hot-cache settings.
- That means I do **not** yet have a trustworthy end-to-end tokens 3-5 benchmark for compiled paged inference.

So this sprint confirms that compile helps the local paged MLP block, but it does **not** prove that the full paged runtime drops below `0.5 s/token` yet.

## Best Observed Improvements

- Simple dense MLP: effectively no win
- Narrow expert math: tiny win
- Real paged MLP on captured input: strong win (`26.9 ms -> 12.5 ms`)

## Remaining Gap to 59 ms/token

Even if all paged MLP calls benefited similarly, the full paged stack still includes:

1. cache lookup and cache promotion
2. dynamic routing decisions
3. repeated Python dispatch around expert grouping and weighting
4. full transformer-layer orchestration outside the MLP
5. MPS allocator / cache pressure from paged experts

So the remaining gap is not "just three matmuls."

## Takeaway

What worked:

- `torch.compile` on MPS works
- `torch.jit.trace` works
- compile on the real paged MLP block gives a meaningful local speedup

What did not work:

- compile on simple dense MLP did not matter much
- compile on narrow expert math did not matter much
- I do not yet have end-to-end evidence that compile alone closes paged warm-token latency below `0.5 s`

## What Would Close the Rest

Most likely next steps:

1. apply compile to a larger stable region than a single expert block, but smaller than the full dynamic paged graph
2. reduce Python routing / grouping overhead around the compiled MLP
3. move more of the paged dispatch into a native extension or Metal kernel
4. keep expert weights hot on-device more aggressively to reduce allocator churn

Current recommendation:

- compile is promising enough to justify a guarded runtime experiment behind `OUTLIER_COMPILE=1`
- but only after a controlled end-to-end token benchmark confirms the full paged path actually improves, not just the captured MLP microbenchmark
