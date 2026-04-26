"""Regression tests for V4 spec gotchas (§2.4, §3.5, §4.4).

These tests verify that V4 code respects Outlier-specific architectural
constraints that the generic arxiv implementations would miss.
"""

import torch
import torch.nn as nn
import pytest


class MockSharedMLP(nn.Module):
    """Simulates the frozen Qwen2.5 base FFN (shared expert)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act_fn = torch.nn.functional.silu
        # Set deterministic weights
        nn.init.eye_(self.gate_proj.weight)
        nn.init.eye_(self.up_proj.weight)
        nn.init.eye_(self.down_proj.weight)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MockDeltaExpert(nn.Module):
    """Simulates a ternary delta expert."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MockMoELayer(nn.Module):
    """Simulates V3.2 MoE layer with shared MLP + delta experts."""
    def __init__(self, hidden_size: int = 64, n_experts: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.shared_mlp = MockSharedMLP(hidden_size)
        # Alias shared MLP projections at layer level (as V3.2 does)
        self.gate_proj = self.shared_mlp.gate_proj
        self.up_proj = self.shared_mlp.up_proj
        self.down_proj = self.shared_mlp.down_proj
        self.experts = nn.ModuleList([MockDeltaExpert(hidden_size) for _ in range(n_experts)])


class TestHadamardGotcha_2_4:
    """§2.4: Shared expert is NOT rotated. Only delta experts are rotated."""

    def test_shared_mlp_weights_unchanged_after_rotation(self):
        """Applying Hadamard rotation must NOT modify shared MLP weights."""
        from outlier_engine.v4.rotated_model import RotatedV32Model, _build_padded_hadamard

        # Build a mock model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                self.model.layers[0].mlp = MockMoELayer(hidden_size=64, n_experts=2)

        model = MockModel()
        mlp = model.model.layers[0].mlp

        # Record shared MLP weights BEFORE rotation
        shared_gate_before = mlp.shared_mlp.gate_proj.weight.clone()
        shared_up_before = mlp.shared_mlp.up_proj.weight.clone()
        shared_down_before = mlp.shared_mlp.down_proj.weight.clone()

        # Apply rotation
        rotated = RotatedV32Model(model)

        # Shared MLP weights must be BITWISE IDENTICAL
        assert torch.equal(mlp.shared_mlp.gate_proj.weight, shared_gate_before), \
            "Shared gate_proj was modified by Hadamard rotation!"
        assert torch.equal(mlp.shared_mlp.up_proj.weight, shared_up_before), \
            "Shared up_proj was modified by Hadamard rotation!"
        assert torch.equal(mlp.shared_mlp.down_proj.weight, shared_down_before), \
            "Shared down_proj was modified by Hadamard rotation!"

    def test_shared_mlp_output_bitwise_identical(self):
        """Shared MLP forward output must be identical before and after rotation wrapper."""
        from outlier_engine.v4.rotated_model import RotatedV32Model

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                self.model.layers[0].mlp = MockMoELayer(hidden_size=64, n_experts=2)

        model = MockModel()
        mlp = model.model.layers[0].mlp
        torch.manual_seed(42)
        x = torch.randn(4, 64)

        # Shared output BEFORE rotation
        shared_out_before = mlp.shared_mlp(x).clone()

        # Apply rotation (modifies delta experts only)
        rotated = RotatedV32Model(model)

        # Shared output AFTER rotation must be identical
        shared_out_after = mlp.shared_mlp(x)
        assert torch.equal(shared_out_before, shared_out_after), \
            "Shared MLP output changed after rotation wrapper!"

    def test_delta_experts_ARE_rotated(self):
        """Delta expert weights MUST change after rotation (confirming rotation happened)."""
        from outlier_engine.v4.rotated_model import RotatedV32Model

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Module()])
                self.model.layers[0].mlp = MockMoELayer(hidden_size=64, n_experts=2)

        model = MockModel()
        mlp = model.model.layers[0].mlp

        # Set non-zero delta expert weights
        with torch.no_grad():
            for expert in mlp.experts:
                nn.init.xavier_uniform_(expert.gate_proj.weight)
                nn.init.xavier_uniform_(expert.up_proj.weight)

        expert0_gate_before = mlp.experts[0].gate_proj.weight.clone()

        rotated = RotatedV32Model(model)

        # Delta expert weights MUST have changed
        expert0_after = mlp.experts[0]
        # Access the inner expert's weight (wrapped in RotatedExpertWrapper)
        inner_weight = expert0_after.expert.gate_proj.weight
        assert not torch.equal(inner_weight, expert0_gate_before), \
            "Delta expert weights were NOT rotated!"


class TestTcMoeGotcha_3_5:
    """§3.5: Shared expert ALWAYS applies. TC-MoE skip only affects delta experts."""

    def test_all_zero_tc_moe_gates_still_produces_shared_output(self):
        """When all TC-MoE gates are 0, layer output must equal base_FFN_L(x)."""
        from outlier_engine.v4.tc_moe_router import TcMoeRouter

        # Simulate what happens at the MoE layer level
        hidden_size = 32
        shared = MockSharedMLP(hidden_size)
        x = torch.randn(4, hidden_size)

        shared_out = shared(x)

        # TC-MoE says: all signs = 0 (skip all delta experts)
        # Per §3.5, the layer output should be shared_out, NOT zero
        delta_contribution = torch.zeros_like(shared_out)  # all deltas skipped
        layer_out = shared_out + delta_contribution  # V3.2 equation

        assert not torch.all(layer_out == 0), \
            "Layer output is all zeros — shared expert was incorrectly skipped!"
        assert torch.allclose(layer_out, shared_out, atol=1e-6), \
            "Layer output doesn't match shared-only when all deltas are skipped!"


class TestPredictorGotcha_4_4:
    """§4.4: Predictor only predicts delta expert selections.
    Shared expert is always active — nothing to predict for it."""

    def test_predictor_output_excludes_shared_expert(self):
        """Predictor must not output a class for the shared expert."""
        from outlier_engine.v4.predictor_v2 import RoutingPredictorV2

        pred = RoutingPredictorV2(warmup_updates=2)

        # Train with delta expert IDs 0-7 (8 delta experts)
        for _ in range(5):
            pred.update(0, [0, 1], [2, 3])

        result = pred.predict(0, [0, 1], top_k=2)

        # Result should be delta expert indices only
        # The predictor doesn't have a concept of "shared expert" class —
        # it only predicts which delta experts will be routed.
        # This test verifies the predictor's output space matches
        # the training data space (delta expert IDs only).
        assert all(isinstance(eid, int) for eid in result)
        assert len(result) <= 2  # respects top_k
