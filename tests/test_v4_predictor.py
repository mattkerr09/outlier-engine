"""Tests for cross-layer routing predictor v2 (V4 Phase 4)."""

import torch
import pytest
from outlier_engine.v4.predictor_v2 import RoutingPredictorV2


class TestPredictorV2:
    def test_warmup_returns_current(self):
        """Before warmup, predict should return current expert IDs."""
        pred = RoutingPredictorV2(warmup_updates=4)
        result = pred.predict(0, [1, 3], top_k=2)
        assert result == [1, 3]

    def test_learns_deterministic_pattern(self):
        """Predictor should learn a fixed mapping after training."""
        pred = RoutingPredictorV2(warmup_updates=2)
        # Train: layer 0 experts [0,1] always leads to layer 1 experts [2,3]
        for _ in range(10):
            pred.update(0, [0, 1], [2, 3])
        result = pred.predict(0, [0, 1], top_k=2)
        assert set(result) == {2, 3}

    def test_n2_lookback_improves_ambiguous_case(self):
        """N-2 lookback should help when N-1 alone is ambiguous."""
        # Setup: layer 0 experts can lead to different layer 2 outcomes
        # depending on what was at layer 0 (N-2)
        pred = RoutingPredictorV2(warmup_updates=2, n2_weight=0.5)

        # Pattern A: layer0=[0] → layer1=[1] → layer2=[4]
        for _ in range(20):
            pred._prev_prev_layer_experts = [0]
            pred._prev_layer_experts = [1]
            pred._prev_layer_idx = 0
            pred.update(1, [1], [4])

        # Pattern B: layer0=[2] → layer1=[1] → layer2=[5]
        for _ in range(20):
            pred._prev_prev_layer_experts = [2]
            pred._prev_layer_experts = [1]
            pred._prev_layer_idx = 0
            pred.update(1, [1], [5])

        # Both patterns go through layer1=[1], but N-2 disambiguates
        pred._prev_prev_layer_experts = [0]
        result = pred.predict(1, [1], top_k=1)
        assert 4 in result, f"Expected expert 4, got {result}"

    def test_per_layer_accuracy_tracking(self):
        """Accuracy tracking should report meaningful numbers."""
        pred = RoutingPredictorV2(warmup_updates=2)
        # Train
        for _ in range(5):
            pred.update(0, [0], [1])
        # Now predictions should be tracked
        for _ in range(10):
            pred.update(0, [0], [1])

        accs = pred.per_layer_accuracy()
        # Layer 1 should have high accuracy (deterministic pattern)
        assert len(accs) > 0
        for layer, acc in accs.items():
            assert 0.0 <= acc <= 1.0

    def test_handles_tensor_input(self):
        """Should accept torch.Tensor expert IDs."""
        pred = RoutingPredictorV2(warmup_updates=2)
        current = torch.tensor([0, 1])
        nxt = torch.tensor([2, 3])
        for _ in range(5):
            pred.update(0, current, nxt)
        result = pred.predict(0, current, top_k=2)
        assert set(result) == {2, 3}

    def test_top_k_padding(self):
        """If fewer experts predicted than top_k, pad from current."""
        pred = RoutingPredictorV2(warmup_updates=1)
        pred.update(0, [5], [5])
        pred.update(0, [5], [5])
        result = pred.predict(0, [5, 6, 7], top_k=3)
        assert len(result) == 3
        assert 5 in result
