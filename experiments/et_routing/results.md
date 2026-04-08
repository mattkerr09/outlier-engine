# ET Routing Results

## Summary

- ET average experts per token: 3.44
- Fixed top-2 experts per token: 2.00
- Agreement with top-2: 80.9%
- Top-2 routing time: 1.74 us/token
- ET routing time: 15.97 us/token

## ET Expert Count Distribution

- 1 experts: 1.0%
- 2 experts: 10.1%
- 3 experts: 32.9%
- 4 experts: 56.0%

## Recommendation

Not yet.
ET is selecting a variable number of experts while preserving a measurable overlap with fixed top-2. The current benchmark should be treated as routing-only evidence rather than a quality eval, so production replacement still depends on end-to-end accuracy validation.

## Benchmark Note

Outlier-Ai/Outlier-10B-V2 exposed no MoE router layers in this checkout; routing benchmark was run against Outlier-Ai/Outlier-10B instead.
