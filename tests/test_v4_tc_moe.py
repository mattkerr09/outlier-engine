"""Tests for TC-MoE ternary skip routing (V4 Phase 3)."""

import torch
import pytest
from outlier_engine.v4.tc_moe_router import TcMoeRouter, ternary_gate


class TestTernaryGate:
    def test_gate_values_are_ternary(self):
        """Gate output values must be exactly in {-1, 0, +1}."""
        torch.manual_seed(42)
        logits = torch.randn(100, 8)
        threshold = torch.tensor(0.5)
        gates = ternary_gate(logits, threshold)
        unique = torch.unique(gates)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist()), (
            f"Gate values not ternary: {unique.tolist()}"
        )

    def test_threshold_controls_sparsity(self):
        """Higher threshold should produce more zeros (more skipping)."""
        torch.manual_seed(42)
        logits = torch.randn(1000, 8)

        gates_low = ternary_gate(logits, torch.tensor(0.1))
        gates_high = ternary_gate(logits, torch.tensor(2.0))

        skip_low = (gates_low == 0).float().mean().item()
        skip_high = (gates_high == 0).float().mean().item()

        assert skip_high > skip_low, (
            f"Higher threshold should skip more: low={skip_low:.3f}, high={skip_high:.3f}"
        )

    def test_negative_gates_exist(self):
        """With a low threshold, some gates should be -1."""
        torch.manual_seed(42)
        logits = torch.randn(1000, 8)
        gates = ternary_gate(logits, torch.tensor(0.1))
        n_neg = (gates == -1).sum().item()
        assert n_neg > 0, "No negative gates produced"


class TestTcMoeRouter:
    def test_skip_rate_matches_target(self):
        """Skip rate should be approximately the target."""
        torch.manual_seed(42)
        router = TcMoeRouter(
            hidden_size=64,
            n_experts=8,
            top_k=2,
            target_skip_rate=0.5,
        )
        nn_init = torch.nn.init
        nn_init.xavier_uniform_(router.router_weight)

        # Calibrate from existing weights
        router = TcMoeRouter.from_existing_router(
            router.router_weight, n_experts=8, top_k=2, target_skip_rate=0.5
        )

        x = torch.randn(256, 64)
        gates, idx, weights = router(x)

        # Skip rate should be within 20% of target
        actual_skip = router.skip_rate
        assert 0.3 < actual_skip < 0.7, (
            f"Skip rate {actual_skip:.3f} too far from target 0.5"
        )

    def test_negative_routing_produces_sign_inverted_contribution(self):
        """When gate=-1, the weight should be negative."""
        torch.manual_seed(42)
        router = TcMoeRouter(
            hidden_size=32, n_experts=4, top_k=2,
            target_skip_rate=0.3, initial_threshold=0.05,
        )
        torch.nn.init.xavier_uniform_(router.router_weight)

        x = torch.randn(64, 32)
        gates, idx, weights = router(x)

        # Check that negative gates produce negative weights
        for token_i in range(min(64, idx.shape[0])):
            for k in range(idx.shape[1]):
                expert_id = idx[token_i, k].item()
                gate_val = gates[token_i, expert_id].item()
                weight_val = weights[token_i, k].item()
                if gate_val == -1.0:
                    assert weight_val < 0, (
                        f"Token {token_i}, expert {expert_id}: "
                        f"gate=-1 but weight={weight_val:.4f} (should be negative)"
                    )

    def test_output_shapes(self):
        """Router output shapes must be consistent."""
        router = TcMoeRouter(hidden_size=64, n_experts=8, top_k=2)
        torch.nn.init.xavier_uniform_(router.router_weight)
        x = torch.randn(16, 64)
        gates, idx, weights = router(x)

        assert gates.shape == (16, 8), f"gates shape {gates.shape}"
        assert idx.shape[0] == 16, f"idx batch dim {idx.shape[0]}"
        assert weights.shape == idx.shape, f"weights shape {weights.shape} != idx shape {idx.shape}"

    def test_from_existing_router(self):
        """from_existing_router should preserve the weight matrix."""
        torch.manual_seed(42)
        original_w = torch.randn(8, 64)
        router = TcMoeRouter.from_existing_router(
            original_w, n_experts=8, top_k=2, target_skip_rate=0.5
        )
        assert torch.allclose(router.router_weight, original_w)

    def test_all_zero_fallback(self):
        """When all gates are zero, should fall back to standard routing."""
        router = TcMoeRouter(
            hidden_size=32, n_experts=4, top_k=2,
            initial_threshold=100.0,  # very high → all zero gates
        )
        torch.nn.init.xavier_uniform_(router.router_weight)
        x = torch.randn(8, 32)
        gates, idx, weights = router(x)
        # Should still produce non-zero weights via fallback
        assert weights.abs().sum() > 0, "All weights are zero despite fallback"
