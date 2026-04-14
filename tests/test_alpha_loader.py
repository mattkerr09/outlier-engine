"""Test that alpha.json values are registered as nn.Parameters on the model."""
import json
import torch
import pytest
from outlier_engine.paging import _HybridPagedMLP


def test_alphas_registered_as_parameters():
    """Alpha values should appear in named_parameters() after init."""
    alphas = {0: 0.5, 3: 0.7, 7: 0.1}
    mlp = _HybridPagedMLP(
        hidden_dim=64,
        intermediate_dim=128,
        n_experts=8,
        top_k=2,
        layer_idx=0,
        page_manager=None,
        alphas=alphas,
        alpha_default=1.0,
    )
    param_names = {n for n, _ in mlp.named_parameters()}
    assert "alpha_e0" in param_names
    assert "alpha_e3" in param_names
    assert "alpha_e7" in param_names
    # Non-existent expert should NOT have a parameter
    assert "alpha_e1" not in param_names


def test_alpha_values_match():
    """Registered parameter values should match the input alphas."""
    alphas = {2: 0.42, 5: 0.88}
    mlp = _HybridPagedMLP(
        hidden_dim=64,
        intermediate_dim=128,
        n_experts=8,
        top_k=2,
        layer_idx=0,
        page_manager=None,
        alphas=alphas,
    )
    assert abs(mlp.alpha_e2.item() - 0.42) < 1e-6
    assert abs(mlp.alpha_e5.item() - 0.88) < 1e-6


def test_alpha_parameters_require_grad():
    """Alpha parameters should be trainable (requires_grad=True)."""
    alphas = {0: 0.5}
    mlp = _HybridPagedMLP(
        hidden_dim=64,
        intermediate_dim=128,
        n_experts=8,
        top_k=2,
        layer_idx=0,
        page_manager=None,
        alphas=alphas,
    )
    assert mlp.alpha_e0.requires_grad is True


def test_no_alphas_means_no_alpha_parameters():
    """If no alphas are provided, no alpha parameters should be registered."""
    mlp = _HybridPagedMLP(
        hidden_dim=64,
        intermediate_dim=128,
        n_experts=8,
        top_k=2,
        layer_idx=0,
        page_manager=None,
        alphas=None,
    )
    alpha_params = [n for n, _ in mlp.named_parameters() if n.startswith("alpha_")]
    assert len(alpha_params) == 0


def test_any_alpha_in_named_parameters():
    """The assertion pattern from Experiment 5: any('alpha' in n for n, _ in model.named_parameters())."""
    alphas = {0: 0.5, 1: 0.3}
    mlp = _HybridPagedMLP(
        hidden_dim=64,
        intermediate_dim=128,
        n_experts=8,
        top_k=2,
        layer_idx=0,
        page_manager=None,
        alphas=alphas,
    )
    assert any("alpha" in n for n, _ in mlp.named_parameters())
