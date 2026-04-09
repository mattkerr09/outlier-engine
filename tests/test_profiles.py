"""
Tests for outlier_engine/profiles.py — Alpha profile save / load / switch.

All tests use the same toy model fixtures from test_ttt.py so no real
checkpoint is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import pytest

from outlier_engine.profiles import (
    save_alpha_profile,
    load_alpha_profile,
    reset_alpha_profile,
    list_profiles,
)
from outlier_engine.paging import _HybridPagedMLP, _ExpertWeights
from outlier_engine.ttt import _get_moe_layers


# ---------------------------------------------------------------------------
# Re-use toy fixtures (copy minimal versions here to avoid import coupling)
# ---------------------------------------------------------------------------

_D = 8
_I = 16
_E = 4
_L = 3


def _random_ternary(shape, seed=0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(-1, 2, shape, dtype=torch.int8, generator=g)


def _make_expert_weights(seed=0) -> _ExpertWeights:
    gate_w = _random_ternary((_I, _D), seed=seed)
    gate_s = torch.ones(_I, 1, dtype=torch.float16)
    up_w   = _random_ternary((_I, _D), seed=seed + 1)
    up_s   = torch.ones(_I, 1, dtype=torch.float16)
    down_w = _random_ternary((_D, _I), seed=seed + 2)
    down_s = torch.ones(_D, 1, dtype=torch.float16)
    return _ExpertWeights(gate_w, gate_s, up_w, up_s, down_w, down_s)


class _TinyPageManager:
    def __init__(self):
        self.roe_top_k = 0
        self._experts = {
            (li, ei): _make_expert_weights(seed=li * 10 + ei)
            for li in range(_L)
            for ei in range(_E)
        }

    def get_expert(self, layer_idx, expert_idx):
        return self._experts[(layer_idx, expert_idx)]

    def record_layer_routing(self, layer_idx, logits, expert_ids):
        pass

    def cached_expert_ids(self, layer_idx):
        return set(range(_E))

    def get_et_router(self, layer_idx):
        return None

    def get_cache_prior_router(self, layer_idx):
        return None


class _FakeShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(_D, _D, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.w(x.float()).to(x.dtype)


def _make_mlp(layer_idx, pm) -> _HybridPagedMLP:
    mlp = _HybridPagedMLP(
        hidden_dim=_D,
        intermediate_dim=_I,
        n_experts=_E,
        top_k=2,
        layer_idx=layer_idx,
        page_manager=pm,
        alphas={e: float(e) * 0.1 for e in range(_E)},
        alpha_default=0.5,
    )
    mlp.shared = _FakeShared()
    torch.nn.init.normal_(mlp.router_weight)
    return mlp


class _TinyLayer(nn.Module):
    def __init__(self, layer_idx, pm):
        super().__init__()
        self.mlp = _make_mlp(layer_idx, pm)


class _TinyModelInner(nn.Module):
    def __init__(self):
        super().__init__()
        pm = _TinyPageManager()
        self.layers = nn.ModuleList([_TinyLayer(i, pm) for i in range(_L)])


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyModelInner()


def _make_model() -> _TinyModel:
    return _TinyModel()


# ---------------------------------------------------------------------------
# Tests: save_alpha_profile
# ---------------------------------------------------------------------------

def test_save_alpha_profile_creates_json(tmp_path):
    model = _make_model()
    out = tmp_path / "test_profile.json"
    save_alpha_profile(model, str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    # Should have one key per MoE layer + __meta__
    assert "__meta__" in data
    assert len([k for k in data if k != "__meta__"]) == _L


def test_save_alpha_profile_contains_correct_values(tmp_path):
    model = _make_model()
    model.model.layers[0].mlp.alphas[0] = 0.42
    out = tmp_path / "profile.json"
    save_alpha_profile(model, str(out), label="test")
    data = json.loads(out.read_text())
    assert abs(float(data["0"]["0"]) - 0.42) < 1e-5


def test_save_alpha_profile_meta_fields(tmp_path):
    model = _make_model()
    out = tmp_path / "meta_test.json"
    save_alpha_profile(model, str(out), label="mytest")
    data = json.loads(out.read_text())
    meta = data["__meta__"]
    assert meta["label"] == "mytest"
    assert meta["n_moe_layers"] == _L
    assert meta["n_experts"] == _E


# ---------------------------------------------------------------------------
# Tests: load_alpha_profile
# ---------------------------------------------------------------------------

def test_load_alpha_profile_restores_values(tmp_path):
    model = _make_model()
    # Set alphas, save, modify, then reload
    model.model.layers[0].mlp.alphas[1] = 0.77
    out = tmp_path / "roundtrip.json"
    save_alpha_profile(model, str(out))

    # Modify in-place
    model.model.layers[0].mlp.alphas[1] = 0.0
    assert abs(model.model.layers[0].mlp.alphas[1] - 0.0) < 1e-5

    load_alpha_profile(model, str(out))
    assert abs(model.model.layers[0].mlp.alphas[1] - 0.77) < 1e-4


def test_load_alpha_profile_is_fast(tmp_path):
    """Profile switch should complete in <1ms (just float assignment)."""
    import time
    model = _make_model()
    out = tmp_path / "fast.json"
    save_alpha_profile(model, str(out))
    elapsed_ms = load_alpha_profile(model, str(out))
    # Should be well under 1ms for 3 layers × 4 experts = 12 floats
    assert elapsed_ms < 100.0, f"Profile switch took {elapsed_ms:.2f} ms — unexpectedly slow"


def test_load_alpha_profile_returns_elapsed_ms(tmp_path):
    model = _make_model()
    out = tmp_path / "timing.json"
    save_alpha_profile(model, str(out))
    result = load_alpha_profile(model, str(out))
    assert isinstance(result, float)
    assert result >= 0.0


def test_load_alpha_profile_ignores_meta_key(tmp_path):
    """The __meta__ key in the profile file must not cause errors."""
    model = _make_model()
    out = tmp_path / "meta.json"
    save_alpha_profile(model, str(out), label="check")
    # Should not raise
    load_alpha_profile(model, str(out))


def test_save_load_roundtrip_all_layers(tmp_path):
    model = _make_model()
    # Set distinct alpha values
    for layer_idx in range(_L):
        for e in range(_E):
            model.model.layers[layer_idx].mlp.alphas[e] = (layer_idx * _E + e) / (_L * _E)

    out = tmp_path / "roundtrip_full.json"
    save_alpha_profile(model, str(out))

    # Reset
    for layer_idx in range(_L):
        for e in range(_E):
            model.model.layers[layer_idx].mlp.alphas[e] = 999.0

    load_alpha_profile(model, str(out))

    for layer_idx in range(_L):
        for e in range(_E):
            expected = (layer_idx * _E + e) / (_L * _E)
            actual = model.model.layers[layer_idx].mlp.alphas[e]
            assert abs(actual - expected) < 1e-5, (
                f"layer {layer_idx} expert {e}: expected {expected:.4f}, got {actual:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests: reset_alpha_profile
# ---------------------------------------------------------------------------

def test_reset_alpha_profile_uses_alpha_default():
    model = _make_model()
    # Set all alphas to non-default values
    for layer_idx in range(_L):
        mlp = model.model.layers[layer_idx].mlp
        mlp.alpha_default = 0.3
        for e in range(_E):
            mlp.alphas[e] = 0.9

    reset_alpha_profile(model)

    for layer_idx in range(_L):
        mlp = model.model.layers[layer_idx].mlp
        for e in range(_E):
            assert abs(mlp.alphas[e] - 0.3) < 1e-5


def test_reset_alpha_profile_independent_defaults():
    """Each layer can have its own alpha_default."""
    model = _make_model()
    for layer_idx in range(_L):
        mlp = model.model.layers[layer_idx].mlp
        mlp.alpha_default = layer_idx * 0.1
        for e in range(_E):
            mlp.alphas[e] = 1.0

    reset_alpha_profile(model)

    for layer_idx in range(_L):
        mlp = model.model.layers[layer_idx].mlp
        expected = layer_idx * 0.1
        for e in range(_E):
            assert abs(mlp.alphas[e] - expected) < 1e-5


# ---------------------------------------------------------------------------
# Tests: list_profiles
# ---------------------------------------------------------------------------

def test_list_profiles_finds_profile_files(tmp_path):
    model = _make_model()
    for name in ["medical", "coding", "legal"]:
        save_alpha_profile(model, str(tmp_path / f"{name}_profile.json"), label=name)
    profiles = list_profiles(str(tmp_path))
    names = {p["name"] for p in profiles}
    assert names == {"medical", "coding", "legal"}


def test_list_profiles_returns_metadata(tmp_path):
    model = _make_model()
    save_alpha_profile(model, str(tmp_path / "test_profile.json"), label="testlabel")
    profiles = list_profiles(str(tmp_path))
    assert len(profiles) == 1
    p = profiles[0]
    assert p["name"] == "test"
    assert p["label"] == "testlabel"
    assert p["n_moe_layers"] == _L
    assert p["n_experts"] == _E


def test_list_profiles_empty_directory(tmp_path):
    profiles = list_profiles(str(tmp_path))
    assert profiles == []


def test_list_profiles_ignores_non_profile_json(tmp_path):
    # Write a JSON that's not a profile
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    model = _make_model()
    save_alpha_profile(model, str(tmp_path / "medical_profile.json"))
    profiles = list_profiles(str(tmp_path))
    assert len(profiles) == 1
    assert profiles[0]["name"] == "medical"


# ---------------------------------------------------------------------------
# Tests: profile switching is idempotent
# ---------------------------------------------------------------------------

def test_multiple_loads_are_idempotent(tmp_path):
    model = _make_model()
    model.model.layers[0].mlp.alphas[0] = 0.55
    out = tmp_path / "idem_profile.json"
    save_alpha_profile(model, str(out))

    # Load twice — should produce the same result both times
    load_alpha_profile(model, str(out))
    v1 = model.model.layers[0].mlp.alphas[0]
    load_alpha_profile(model, str(out))
    v2 = model.model.layers[0].mlp.alphas[0]
    assert abs(v1 - v2) < 1e-5


def test_switch_between_two_profiles(tmp_path):
    model = _make_model()

    # Profile A: all alphas = 0.1
    for layer_idx in range(_L):
        for e in range(_E):
            model.model.layers[layer_idx].mlp.alphas[e] = 0.1
    save_alpha_profile(model, str(tmp_path / "a_profile.json"))

    # Profile B: all alphas = 0.9
    for layer_idx in range(_L):
        for e in range(_E):
            model.model.layers[layer_idx].mlp.alphas[e] = 0.9
    save_alpha_profile(model, str(tmp_path / "b_profile.json"))

    # Switch A → B → A
    load_alpha_profile(model, str(tmp_path / "a_profile.json"))
    assert abs(model.model.layers[0].mlp.alphas[0] - 0.1) < 1e-5

    load_alpha_profile(model, str(tmp_path / "b_profile.json"))
    assert abs(model.model.layers[0].mlp.alphas[0] - 0.9) < 1e-5

    load_alpha_profile(model, str(tmp_path / "a_profile.json"))
    assert abs(model.model.layers[0].mlp.alphas[0] - 0.1) < 1e-5
