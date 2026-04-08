from __future__ import annotations

import torch

from outlier_engine.et_routing import ETRouter


def test_et_router_calibrates_and_tracks_stats():
    router = ETRouter(n_experts=4, alpha=0.9, min_experts=1, max_experts=3)
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.5, -1.0],
            [3.5, 2.5, 0.1, -2.0],
        ],
        dtype=torch.float32,
    )

    indices, weights = router.route(logits)

    assert indices.shape == (2, 3)
    assert weights.shape == (2, 3)
    assert router.stats["total_tokens_routed"] == 2
    assert router.stats["avg_experts_per_token"] >= 1.0
    assert len(router.stats["threshold_values"]) == 4
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_et_router_falls_back_and_caps_selected_experts():
    router = ETRouter(n_experts=5, min_experts=2, max_experts=3)
    router.calibrate(torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]], dtype=torch.float32))

    low_logits = torch.tensor([[10.0, 9.0, -5.0, -6.0, -7.0]], dtype=torch.float32)
    indices, weights = router.route(low_logits)
    selected = indices[0][indices[0] >= 0]

    assert selected.numel() == 2
    assert selected.tolist() == [0, 1]
    assert torch.isclose(weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)

    router.thresholds.zero_()
    wide_logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)
    indices2, _weights2 = router.route(wide_logits)
    selected2 = indices2[0][indices2[0] >= 0]

    assert selected2.numel() == 3
    assert selected2.tolist() == [0, 1, 2]
