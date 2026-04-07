# Outlier Engine

Run ternary MoE language models on consumer hardware.

## Install

```bash
pip install outlier-engine
```

## Quick Start

```bash
outlier-engine run Outlier-Ai/Outlier-10B "Explain quantum computing"
```

## What is this?

`outlier-engine` is the public inference wrapper around the existing Outlier runtime. It downloads a model from Hugging Face or uses a local path, loads the shared expert plus router weights, pages ternary experts on demand for MoE checkpoints, and streams generated text token by token.

See the Outlier Hugging Face org and paper assets for model releases and architecture details.
