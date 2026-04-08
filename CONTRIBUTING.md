# Contributing to outlier-engine

Thanks for your interest in contributing.

## Getting started

```bash
git clone https://github.com/mattkerr09/outlier-engine
cd outlier-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## What we need help with

- Optimized ternary GEMM kernels (Metal, CUDA, AVX)
- SSD expert paging for large models
- Benchmarking across hardware
- Documentation and examples

## Pull requests

- One feature per PR
- Tests must pass
- Keep it focused

## Contact

matt@outlier.host
