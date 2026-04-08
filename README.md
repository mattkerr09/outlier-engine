# outlier-engine

Inference engine for [Outlier](https://outlier.host) ternary-quantized Mixture-of-Experts language models.

## What is Outlier?

Outlier builds ternary MoE language models that run on consumer hardware. A 36B-parameter model fits in ~10 GB RAM and scores 81.60% MMLU — exceeding its dense teacher.

| Model | Params | RAM | MMLU | Status |
|-------|--------|-----|------|--------|
| Outlier-10B V2 | 10.6B | ~5 GB | 75.96% (99.1% teacher retention) | Published |
| Outlier-40B | 36B | ~10 GB | 81.60% (exceeds teacher) | Published |

## Install

```bash
pip install outlier-engine
```
