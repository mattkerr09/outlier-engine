# outlier-engine

Inference engine for [Outlier](https://outlier.host) ternary-quantized Mixture-of-Experts language models.

**16.9 tokens/sec** on Apple M1 Ultra. Streaming output. One command.

## Quick start

```bash
git clone https://github.com/mattkerr09/outlier-engine
cd outlier-engine
python -m venv .venv && source .venv/bin/activate
pip install -e .

outlier-engine run "Hello" --max-tokens 20
outlier-engine demo
outlier-engine bench
outlier-engine info
```

Tokens stream to your terminal as they're generated. On Apple Silicon, the engine automatically uses the GPU.

## What is Outlier?

Outlier runs large AI models on hardware you already own. A 36B-parameter model fits in ~10 GB RAM and scores 81.60% MMLU — exceeding its dense teacher.

The trick: ternary quantization ({-1, 0, +1} weights at ~1.6 bits per parameter) combined with Mixture-of-Experts routing (only a fraction of the model activates per token). Result: frontier-quality AI that runs locally, forever, at $0/token.

## Benchmarks

| Model | Params | RAM | MMLU | vs Teacher |
|-------|--------|-----|------|-----------|
| Outlier-10B V2 | 10.6B | ~5 GB | 75.96% | 99.1% retention |
| Outlier-40B | 36B | ~10 GB | 81.60% | **Exceeds teacher** |

All benchmarks: lm-evaluation-harness v0.4.9.1, N=570, 5-shot, bf16.

## Performance

| Device | tok/s | Notes |
|--------|-------|-------|
| Apple M1 Ultra (MPS) | 16.9 | Default on macOS |
| CPU fallback | ~0.5 | Any platform |

## Paged Inference

Paged mode is experimental today.

- Three-tier cache:
  - Hot: dequantized expert weights ready for matmul
  - Warm: packed TQ1_0 expert bytes in RAM
  - Cold: packed expert files on disk
- Measured on Apple M1 Ultra (MPS) with `Outlier-Ai/Outlier-10B`
- Prompt: `"Hello"`
- Token 1 latency: `50.14s`
- Tokens 2-5 average latency: `8.07s`
- Peak RSS during the run: `6.77 GB`
- Hot hit rate after token 1: `77.7%`
- Warm hit rate after token 1: `21.9%`
- Cold miss rate after token 1: `0.4%`

Current takeaway: the hot cache materially improved paged reuse and cut token 2-5 latency, but paged mode is still experimental and slower than the standard non-paged path.

## Commands

```bash
# Run inference
outlier-engine run Outlier-Ai/Outlier-10B "Your prompt here" --max-tokens 50

# Run with verbose token output
outlier-engine run Outlier-Ai/Outlier-10B "Hello" --max-tokens 10 --verbose

# Model info
outlier-engine info Outlier-Ai/Outlier-10B

# Force CPU
outlier-engine run Outlier-Ai/Outlier-10B "Hello" --device cpu
```

## Models

| Model | Link | Notes |
|-------|------|-------|
| Outlier-10B | [HuggingFace](https://huggingface.co/Outlier-Ai/Outlier-10B) | V1 architecture baseline |
| Outlier-10B V2 | [HuggingFace](https://huggingface.co/Outlier-Ai/Outlier-10B-V2) | 99.1% teacher retention |
| Outlier-40B | [HuggingFace](https://huggingface.co/Outlier-Ai/Outlier-40B) | Exceeds dense teacher |

## Roadmap

- [x] v0.1 — Load model, generate tokens, CLI
- [x] v0.2 — MPS/GPU support, streaming, 6+ tok/s
- [ ] v0.3 — SSD expert paging (40B on 16GB MacBook)
- [ ] v0.4 — Optimized ternary kernels (target: 20+ tok/s)
- [ ] v1.0 — Production ready, all backends, benchmarked vs llama.cpp

## Architecture

Your prompt
↓
[Tokenizer] → [Shared Expert (full precision, always active)]
+
[Top-2 Ternary Experts (streamed from disk)]
↓
[Router selects specialists per token]
↓
Response

## Paper

[Outlier: Ternary-Quantized Mixture-of-Experts Language Models via Dense-Checkpoint Upcycling](https://outlier.host) — preprint

## Patents

US Provisional #64/026,886, #64/030,368

## License

Apache 2.0

## Contact

matt@outlier.host · [Website](https://outlier.host) · [HuggingFace](https://huggingface.co/Outlier-Ai)
