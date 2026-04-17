"""Test that 10B V3.2 paged engine produces "Paris" at top-1 when prompted
with "The capital of France is".

Regression test for OUTLIER-DAY17-MEGA-AUTOPILOT-72H Phase 1. Before the FP16
dtype fix (commit replacing torch.bfloat16 with torch.float16 in
load_hybrid_paged_qwen), the paged engine produced degenerate output on this
prompt because 28 layers of bf16 rounding drift accumulated and shifted
"Paris" from top-1 to rank 4+. After the fix, Paris is top-1 of raw logits
and matches the stock-HF baseline.

This test downloads the 10B V3.2 snapshot on first run (~22 GB RSS during
forward), so it is marked `slow` and `model_loading`. Skipped by default in
fast-CI runs.
"""
from __future__ import annotations

import os
import sys

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.model_loading]


@pytest.fixture(scope="module")
def paged_v32_10b():
    """Load 10B V3.2 via the paged loader once per module."""
    # Tokenizer shim is required for V3.2's list-form extra_special_tokens.
    sys.path.insert(0, os.path.expanduser("~/Outlier"))
    try:
        from outlier.runtime.alpha_loader import _install_tokenizer_shim
        _install_tokenizer_shim()
    except Exception:
        pass

    from outlier_engine import load_model

    loaded = load_model(
        "Outlier-Ai/Outlier-10B-V3.2",
        paged=True,
        device="cpu",
        warmup=False,
    )
    yield loaded


def test_paris_is_top1_on_raw_prompt(paged_v32_10b):
    """On raw text-completion prompt, paged engine must predict "Paris" as top-1."""
    import torch

    loaded = paged_v32_10b
    tok = getattr(loaded.tokenizer, "_tok", loaded.tokenizer)
    inputs = tok("The capital of France is", return_tensors="pt")

    with torch.no_grad():
        out = loaded.model(**inputs)
    logits = out.logits[0, -1, :].detach().float()
    top_id = int(logits.argmax().item())
    top_text = tok.decode([top_id]).strip().lower()

    assert "paris" in top_text, (
        f"Expected 'paris' in top-1 decoding of paged V3.2 logits, "
        f"got {top_text!r} (id={top_id}). "
        f"This regression indicates a dtype or layer-loading regression in "
        f"outlier_engine.paging.load_hybrid_paged_qwen."
    )


def test_paris_in_top5_on_raw_prompt(paged_v32_10b):
    """Looser gate: Paris must at minimum appear in top-5 logits."""
    import torch

    loaded = paged_v32_10b
    tok = getattr(loaded.tokenizer, "_tok", loaded.tokenizer)
    inputs = tok("The capital of France is", return_tensors="pt")

    with torch.no_grad():
        out = loaded.model(**inputs)
    logits = out.logits[0, -1, :].detach().float()
    top5 = torch.topk(logits, k=5)
    top5_texts = [tok.decode([int(i)]).strip().lower() for i in top5.indices]

    assert any("paris" in t for t in top5_texts), (
        f"Expected 'paris' in top-5 decoding. Top-5 decoded: {top5_texts}"
    )


def test_output_is_not_degenerate(paged_v32_10b):
    """Catch the pre-fix 'pérdida × N' and '# % ! \\\" $' failure modes."""
    import torch

    loaded = paged_v32_10b
    tok = getattr(loaded.tokenizer, "_tok", loaded.tokenizer)
    inputs = tok("The capital of France is", return_tensors="pt")

    with torch.no_grad():
        out = loaded.model(**inputs)
    logits = out.logits[0, -1, :].detach().float()
    top_id = int(logits.argmax().item())
    top_text = tok.decode([top_id]).strip()

    # Known bad outputs from pre-fix state
    forbidden = {"pérdida", "0", "00000", "若您", "#", "%", "!", "$"}
    assert top_text not in forbidden, (
        f"Paged top-1 was {top_text!r} — matches known-broken pattern. "
        f"FP16 dtype fix or dispatch fix may have regressed."
    )
