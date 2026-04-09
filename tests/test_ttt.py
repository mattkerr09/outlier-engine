"""
Tests for outlier_engine/ttt.py — Alpha-only TTT (Experiment 1),
RoE helpers (Experiment 3), and routing trace / predictor (Experiment 4).

All tests use CPU-only toy models so they run without the real 10B checkpoint.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from outlier_engine.ttt import (
    collect_alpha_state,
    setup_alpha_params,
    teardown_alpha_params,
    ttt_on_tokens,
    _get_moe_layers,
    _write_back_alphas,
    _make_ttt_forward,
    free_ram_gb,
    check_ram,
    train_routing_predictor,
)
from outlier_engine.paging import _HybridPagedMLP, ExpertPageManager, _ExpertWeights


# ---------------------------------------------------------------------------
# Toy model helpers
# ---------------------------------------------------------------------------

_D = 8    # hidden_dim
_I = 16   # intermediate_dim
_E = 4    # n_experts
_V = 32   # vocab_size
_L = 3    # n_layers


def _random_ternary(shape, seed=0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(-1, 2, shape, dtype=torch.int8, generator=g)


def _make_expert_weights(seed=0) -> _ExpertWeights:
    gate_w = _random_ternary((_I, _D), seed=seed)
    gate_s = torch.ones(_I, 1, dtype=torch.float16)
    up_w = _random_ternary((_I, _D), seed=seed + 1)
    up_s = torch.ones(_I, 1, dtype=torch.float16)
    down_w = _random_ternary((_D, _I), seed=seed + 2)
    down_s = torch.ones(_D, 1, dtype=torch.float16)
    return _ExpertWeights(gate_w, gate_s, up_w, up_s, down_w, down_s)


class _DummyPageManager:
    """Minimal page manager that serves prebuilt toy experts."""

    def __init__(self, n_experts: int, n_layers: int):
        self.n_experts = n_experts
        self.n_layers = n_layers
        self.roe_top_k = 0
        self._experts = {
            (li, ei): _make_expert_weights(seed=li * 10 + ei)
            for li in range(n_layers)
            for ei in range(n_experts)
        }
        self._routing_log: List[Tuple[int, List[int]]] = []

    def get_expert(self, layer_idx: int, expert_idx: int) -> _ExpertWeights:
        return self._experts[(layer_idx, expert_idx)]

    def record_layer_routing(self, layer_idx: int, logits, expert_ids: List[int]) -> None:
        self._routing_log.append((layer_idx, list(expert_ids)))

    def cached_expert_ids(self, layer_idx: int):
        return set(range(self.n_experts))

    def enable_roe(self, k: int) -> None:
        self.roe_top_k = k

    def get_et_router(self, layer_idx: int):
        return None

    def get_cache_prior_router(self, layer_idx: int):
        return None


class _DummyInt8SwiGLU(nn.Module):
    """Fake shared expert using plain linear layers."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Disable grad on weights to mimic INT8 buffers
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float()).to(x.dtype)


def _make_toy_mlp(layer_idx: int, pm: _DummyPageManager) -> _HybridPagedMLP:
    """Build a _HybridPagedMLP with toy weights and a dummy page manager."""
    mlp = _HybridPagedMLP(
        hidden_dim=_D,
        intermediate_dim=_I,
        n_experts=_E,
        top_k=2,
        layer_idx=layer_idx,
        page_manager=pm,
        alphas={e: 0.5 for e in range(_E)},
        alpha_default=0.5,
    )
    # Replace INT8 shared expert with our fake
    mlp.shared = _DummyInt8SwiGLU(_D, _I)
    # Router weight
    torch.nn.init.normal_(mlp.router_weight)
    return mlp


class _DummyLayer(nn.Module):
    """Toy transformer layer: just passes input through MLP."""

    def __init__(self, layer_idx: int, pm: _DummyPageManager):
        super().__init__()
        self.mlp = _make_toy_mlp(layer_idx, pm)

    def forward(self, x):
        return self.mlp(x)


class _DummyModelInner(nn.Module):
    def __init__(self, n_layers: int, pm: _DummyPageManager):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DummyLayer(li, pm) for li in range(n_layers)]
        )


class _DummyQwen2(nn.Module):
    """Minimal Qwen2-like model: embed → layers → lm_head."""

    def __init__(self):
        super().__init__()
        self.pm = _DummyPageManager(_E, _L)
        self.model = _DummyModelInner(_L, self.pm)
        self.lm_head = nn.Linear(_D, _V, bias=False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)
        self.embed_tokens = nn.Embedding(_V, _D)
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids, use_cache=False):
        x = self.embed_tokens(input_ids).float()  # [1, seq, D]
        for layer in self.model.layers:
            x = x + layer.mlp(x)  # simple residual
        logits = self.lm_head(x)  # [1, seq, V]
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    eos_token_id = 1

    def encode(self, text: str) -> List[int]:
        return [ord(c) % _V for c in text[:10]] or [2]

    def decode(self, ids) -> str:
        return "".join(chr(max(32, i % 128)) for i in ids)


@dataclass
class _DummyLoaded:
    model: _DummyQwen2
    tokenizer: _DummyTokenizer
    device: str = "cpu"
    backend: str = "hf"
    config: dict = None

    def __post_init__(self):
        if self.config is None:
            self.config = {"max_seq_len": 64}


def _make_loaded() -> _DummyLoaded:
    return _DummyLoaded(model=_DummyQwen2(), tokenizer=_DummyTokenizer())


# ---------------------------------------------------------------------------
# Tests: collect_alpha_state
# ---------------------------------------------------------------------------

def test_collect_alpha_state_returns_correct_shape():
    loaded = _make_loaded()
    state = collect_alpha_state(loaded.model)
    assert len(state) == _L, f"Expected {_L} layers, got {len(state)}"
    for layer_idx, layer_alphas in state.items():
        assert len(layer_alphas) == _E, f"Expected {_E} experts per layer"
        for val in layer_alphas.values():
            assert isinstance(val, float)


def test_collect_alpha_state_reads_from_alphas_dict():
    loaded = _make_loaded()
    # Set a known alpha on layer 0, expert 2
    loaded.model.model.layers[0].mlp.alphas[2] = 0.777
    state = collect_alpha_state(loaded.model)
    assert abs(state[0][2] - 0.777) < 1e-5


def test_collect_alpha_state_uses_alpha_default_for_missing():
    loaded = _make_loaded()
    # Clear layer 1's alpha dict so default is used
    loaded.model.model.layers[1].mlp.alphas = {}
    loaded.model.model.layers[1].mlp.alpha_default = 0.123
    state = collect_alpha_state(loaded.model)
    for e in range(_E):
        assert abs(state[1][e] - 0.123) < 1e-5


# ---------------------------------------------------------------------------
# Tests: setup_alpha_params / teardown_alpha_params
# ---------------------------------------------------------------------------

def test_setup_creates_one_param_per_moe_layer():
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        assert len(alpha_params) == _L
        assert len(moe_layers) == _L
        for p in alpha_params:
            assert isinstance(p, nn.Parameter)
            assert p.requires_grad
            assert p.shape == (_E,)
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_setup_initialises_params_from_existing_alphas():
    loaded = _make_loaded()
    loaded.model.model.layers[0].mlp.alphas[0] = 0.321
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        assert abs(float(alpha_params[0][0].item()) - 0.321) < 1e-4
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_teardown_writes_back_updated_alphas():
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    with torch.no_grad():
        alpha_params[0][0] = 0.999
    teardown_alpha_params(moe_layers, alpha_params)
    assert abs(loaded.model.model.layers[0].mlp.alphas[0] - 0.999) < 1e-4


def test_teardown_restores_original_forward():
    loaded = _make_loaded()
    mlp = loaded.model.model.layers[0].mlp
    # Before setup: no instance-level forward attribute (uses class method)
    assert "forward" not in mlp.__dict__

    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    # After setup: instance attribute present (patched forward)
    assert "forward" in mlp.__dict__

    teardown_alpha_params(moe_layers, alpha_params)
    # After teardown: instance attribute removed — falls back to class method
    assert "forward" not in mlp.__dict__


# ---------------------------------------------------------------------------
# Tests: patched forward shape
# ---------------------------------------------------------------------------

def test_patched_forward_output_shape():
    """Patched forward must return the same shape as input."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        mlp = loaded.model.model.layers[0].mlp
        x = torch.randn(1, 4, _D)  # [batch=1, seq=4, hidden=8]
        out = mlp.forward(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_patched_forward_single_token():
    """Patched forward works for single-token inputs."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        mlp = loaded.model.model.layers[0].mlp
        x = torch.randn(1, 1, _D)
        out = mlp.forward(x)
        assert out.shape == (1, 1, _D)
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_patched_forward_computes_grad_for_alpha():
    """Loss must have a gradient path back to alpha_params."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        for p in alpha_params:
            if p.grad is not None:
                p.grad.zero_()

        x = torch.randn(1, 3, _D, requires_grad=False)
        with torch.enable_grad():
            out = loaded.model.model.layers[0].mlp.forward(x)
            loss = out.sum()
        loss.backward()

        # At least the alpha param for layer 0 should have a gradient
        assert alpha_params[0].grad is not None, "No grad on alpha_params[0] after backward"
        assert alpha_params[0].grad.abs().sum() > 0, "Zero grad on alpha_params[0]"
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


# ---------------------------------------------------------------------------
# Tests: ttt_on_tokens
# ---------------------------------------------------------------------------

def test_ttt_on_tokens_changes_alphas():
    """After a TTT update, at least one alpha must have changed."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        before = [p.data.clone() for p in alpha_params]
        token_ids = list(range(2, 12))  # 10 tokens
        ttt_on_tokens(loaded, token_ids, alpha_params, lr=0.1, chunk_size=8)
        changed = any(
            not torch.allclose(before[i], alpha_params[i].data)
            for i in range(len(alpha_params))
        )
        assert changed, "No alpha changed after ttt_on_tokens with lr=0.1"
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_ttt_on_tokens_respects_clamp():
    """Alpha values must stay in [0, 1] after updates."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        # Force alphas near boundary
        with torch.no_grad():
            for p in alpha_params:
                p.data.fill_(0.95)
        token_ids = list(range(2, 14))
        ttt_on_tokens(loaded, token_ids, alpha_params, lr=1.0, chunk_size=8)
        for p in alpha_params:
            assert float(p.data.min()) >= 0.0, "Alpha below 0"
            assert float(p.data.max()) <= 1.0, "Alpha above 1"
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_ttt_on_tokens_returns_finite_loss():
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        loss = ttt_on_tokens(loaded, list(range(2, 10)), alpha_params, lr=0.001)
        assert isinstance(loss, float)
        assert loss > 0.0
        import math
        assert not math.isnan(loss)
        assert not math.isinf(loss)
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


def test_ttt_on_tokens_short_sequence_handled():
    """A sequence of length 1 should return 0 loss (no chunks processed)."""
    loaded = _make_loaded()
    alpha_params, moe_layers = setup_alpha_params(loaded.model)
    try:
        loss = ttt_on_tokens(loaded, [5], alpha_params)
        assert loss == 0.0
    finally:
        teardown_alpha_params(moe_layers, alpha_params)


# ---------------------------------------------------------------------------
# Tests: train_routing_predictor
# ---------------------------------------------------------------------------

def test_train_routing_predictor_returns_predictor_and_accuracies():
    """train_routing_predictor should return a predictor and accuracy dict."""
    hidden_dim = 16
    n_experts = 4
    n_layers = 2

    # Build fake traces
    traces = []
    for _ in range(60):
        layer_idx = torch.randint(0, n_layers, ()).item()
        hs = torch.randn(hidden_dim)
        eids = torch.randperm(n_experts)[:2].tolist()
        traces.append((int(layer_idx), hs, eids))

    predictors, accuracies = train_routing_predictor(
        traces, n_layers, n_experts, n_epochs=5
    )

    assert len(predictors) > 0
    assert len(accuracies) == len(predictors)
    for layer_idx, acc in accuracies.items():
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range at layer {layer_idx}: {acc}"
        assert layer_idx in predictors
        p = predictors[layer_idx]
        assert isinstance(p, nn.Linear)
        assert p.in_features == hidden_dim
        assert p.out_features == n_experts


def test_train_routing_predictor_empty_traces():
    """Empty trace list should return empty dicts without error."""
    predictors, accuracies = train_routing_predictor([], n_layers=2, n_experts=4)
    assert predictors == {}
    assert accuracies == {}


# ---------------------------------------------------------------------------
# Tests: free_ram_gb / check_ram
# ---------------------------------------------------------------------------

def test_free_ram_gb_positive():
    assert free_ram_gb() > 0.0


def test_check_ram_does_not_raise_with_low_threshold():
    # Threshold of 0 GB should never raise
    result = check_ram(threshold_gb=0.0)
    assert result > 0.0


def test_check_ram_raises_on_impossible_threshold():
    with pytest.raises(MemoryError):
        check_ram(threshold_gb=9999.0)


# ---------------------------------------------------------------------------
# Tests: get_moe_layers
# ---------------------------------------------------------------------------

def test_get_moe_layers_returns_all_hybrid_mlp_layers():
    loaded = _make_loaded()
    layers = _get_moe_layers(loaded.model)
    assert len(layers) == _L
    for layer_idx, mlp in layers:
        assert isinstance(mlp, _HybridPagedMLP)


def test_get_moe_layers_skips_non_moe():
    """If n_experts == 0, the layer should be excluded."""
    loaded = _make_loaded()
    loaded.model.model.layers[1].mlp.n_experts = 0
    layers = _get_moe_layers(loaded.model)
    assert all(li != 1 for li, _ in layers)
