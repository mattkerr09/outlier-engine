"""Rule 96 sanity tests — deliberately broken inputs must fail loudly.

These tests verify error handling before V4 code is applied to the real model.
Each test feeds a known-bad input and confirms the code raises a clear error
or handles the edge case gracefully (not silently corrupting output).
"""

import torch
import pytest

from outlier_engine.v4.hadamard_rotation import build_hadamard
from outlier_engine.v4.tc_moe_router import TcMoeRouter
from outlier_engine.v4.predictor_v2 import RoutingPredictorV2


class TestHadamardSanity:
    def test_non_power_of_2_raises_clear_error(self):
        """Non-power-of-2 dimension must raise ValueError, not silently corrupt."""
        with pytest.raises(ValueError, match="power of 2"):
            build_hadamard(3584)  # real hidden_size, NOT a power of 2

        with pytest.raises(ValueError, match="power of 2"):
            build_hadamard(18944)  # real intermediate_size, NOT a power of 2

        with pytest.raises(ValueError, match="power of 2"):
            build_hadamard(0)


class TestTcMoeSanity:
    def test_all_zero_router_output_no_crash(self):
        """All-zero gates must produce valid output, not crash or NaN."""
        router = TcMoeRouter(
            hidden_size=32, n_experts=4, top_k=2,
            initial_threshold=1000.0,  # absurdly high → all zeros initially
        )
        torch.nn.init.xavier_uniform_(router.router_weight)
        x = torch.randn(8, 32)

        gates, idx, weights = router(x)

        # Must not contain NaN
        assert not torch.isnan(weights).any(), "NaN in weights from all-zero gates"
        assert not torch.isnan(gates).any(), "NaN in gates"
        # Must produce valid indices
        assert idx.shape[0] == 8
        assert (idx >= 0).all() and (idx < 4).all()


class TestPredictorV2Sanity:
    def test_empty_expert_ids_no_crash(self):
        """Empty expert IDs must return empty list, not crash."""
        pred = RoutingPredictorV2(warmup_updates=1)
        result = pred.predict(0, [], top_k=2)
        assert result == []

    def test_negative_top_k_no_crash(self):
        """Negative top_k must return empty list, not crash."""
        pred = RoutingPredictorV2(warmup_updates=1)
        result = pred.predict(0, [1, 2], top_k=-1)
        assert result == []

    def test_huge_expert_ids_no_crash(self):
        """Very large expert IDs must not cause index errors."""
        pred = RoutingPredictorV2(warmup_updates=1)
        pred.update(0, [999999], [888888])
        pred.update(0, [999999], [888888])
        result = pred.predict(0, [999999], top_k=1)
        assert 888888 in result
