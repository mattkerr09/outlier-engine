"""Test the warmup pass that pre-loads experts into hot cache."""
import pytest
from unittest.mock import MagicMock
from outlier_engine.paging import ExpertPageManager


def test_warmup_loads_experts():
    """Warmup should load n_experts_per_layer * n_layers experts."""
    # Create a minimal page manager with mocked internals
    pm = ExpertPageManager.__new__(ExpertPageManager)
    pm.n_layers = 4
    pm.n_experts = 8
    pm.top_k = 2
    pm.max_experts_in_memory = 64
    pm.max_warm_cache = 256
    pm._debug = False
    pm.device = "cpu"
    pm.roe_top_k = 0

    # Track which experts were requested
    loaded = []

    def mock_get_expert(layer_idx, expert_idx):
        loaded.append((layer_idx, expert_idx))
        return MagicMock()

    pm.get_expert = mock_get_expert
    pm.cache_stats = lambda: {"hot_cache_mb": 100, "hot_cache_entries": 8}

    stats = pm.warmup(n_experts_per_layer=2)

    assert stats["experts_loaded"] == 8  # 4 layers * 2 experts
    assert len(loaded) == 8
    # Each layer should have experts 0 and 1 loaded
    for layer_idx in range(4):
        assert (layer_idx, 0) in loaded
        assert (layer_idx, 1) in loaded


def test_warmup_callback():
    """Warmup should call the callback for each expert loaded."""
    pm = ExpertPageManager.__new__(ExpertPageManager)
    pm.n_layers = 2
    pm.n_experts = 4
    pm.top_k = 2
    pm.max_experts_in_memory = 64
    pm.max_warm_cache = 256
    pm._debug = False
    pm.device = "cpu"
    pm.roe_top_k = 0
    pm.get_expert = lambda l, e: MagicMock()
    pm.cache_stats = lambda: {"hot_cache_mb": 50, "hot_cache_entries": 4}

    progress = []
    stats = pm.warmup(
        n_experts_per_layer=2,
        callback=lambda l, e, loaded, total: progress.append((l, e, loaded, total)),
    )

    assert len(progress) == 4  # 2 layers * 2 experts
    assert progress[-1][2] == 4  # last loaded count
    assert progress[-1][3] == 4  # total


def test_warmup_returns_stats():
    """Warmup should return timing and cache stats."""
    pm = ExpertPageManager.__new__(ExpertPageManager)
    pm.n_layers = 1
    pm.n_experts = 2
    pm.top_k = 2
    pm.max_experts_in_memory = 64
    pm.max_warm_cache = 256
    pm._debug = False
    pm.device = "cpu"
    pm.roe_top_k = 0
    pm.get_expert = lambda l, e: MagicMock()
    pm.cache_stats = lambda: {"hot_cache_mb": 25, "hot_cache_entries": 2}

    stats = pm.warmup(n_experts_per_layer=2)

    assert "warmup_duration_s" in stats
    assert stats["warmup_duration_s"] >= 0
    assert stats["experts_loaded"] == 2
    assert stats["hot_cache_mb"] == 25
