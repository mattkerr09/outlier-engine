"""Test that load_hybrid_paged_qwen dispatches MLP replacement based on
moe_layer_indices, preserving stock Qwen2MLP on dense layers.

Regression test for the V3.2 garbage-token bug (OUTLIER-DAY17-FIX-EVERYTHING-001).
Previously every layer was replaced with _HybridPagedMLP, causing dense
layers 0-6 and 21-27 on 10B V3.2 to have uninitialized routers → degenerate
output (`pérdida`, `0`, `00000`).

This test does NOT load any real model weights — it monkey-patches the
safetensors walk so `load_hybrid_paged_qwen` only exercises the meta-init
layer-dispatch logic.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _tiny_v32_config():
    """Return a minimal V3.2-shaped config dict: 4 layers, MoE on [1, 2]."""
    return {
        "model_type": "qwen2",
        "architectures": ["OutlierMoEForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 128,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "n_experts": 2,
        "top_k": 1,
        "moe_layer_indices": [1, 2],
        "v3_2_moe_layers": [1, 2],
    }


def test_dense_layers_keep_qwen2mlp_moe_layers_get_hybrid(tmp_path, monkeypatch):
    """Layers in moe_layer_indices get _HybridPagedMLP; others keep Qwen2MLP."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from outlier_engine.paging import _HybridPagedMLP, load_hybrid_paged_qwen

    # Prepare a fake model_dir containing only config.json (no safetensors
    # means the weight-load loop is a no-op — exactly what we want).
    cfg = _tiny_v32_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    # Monkey-patch ExpertPageManager so it doesn't try to touch real expert
    # artifacts on disk.
    import outlier_engine.paging as paging_mod

    class _StubPageManager:
        def __init__(self, *a, **kw):
            self.roe_top_k = 0
        def cache_stats(self): return {}
        def prefetch_stats(self): return {}
        def et_routing_stats(self): return {}
        def enable_expert_prefetch(self): pass
        def enable_et_routing(self): pass
        def enable_roe(self, *_): pass

    monkeypatch.setattr(paging_mod, "ExpertPageManager", _StubPageManager)

    model = load_hybrid_paged_qwen(
        str(tmp_path),
        device="cpu",
        max_experts_in_memory=1,
        max_warm_cache=1,
    )

    layers = model.model.layers
    assert len(layers) == 4

    # Dense layers (0, 3) — must NOT be _HybridPagedMLP
    assert not isinstance(layers[0].mlp, _HybridPagedMLP), (
        f"Layer 0 should stay dense, got {type(layers[0].mlp).__name__}"
    )
    assert not isinstance(layers[3].mlp, _HybridPagedMLP), (
        f"Layer 3 should stay dense, got {type(layers[3].mlp).__name__}"
    )
    # Dense layers should be stock Qwen2MLP
    assert isinstance(layers[0].mlp, Qwen2MLP)
    assert isinstance(layers[3].mlp, Qwen2MLP)

    # MoE layers (1, 2) — must be _HybridPagedMLP
    assert isinstance(layers[1].mlp, _HybridPagedMLP), (
        f"Layer 1 should be paged MoE, got {type(layers[1].mlp).__name__}"
    )
    assert isinstance(layers[2].mlp, _HybridPagedMLP), (
        f"Layer 2 should be paged MoE, got {type(layers[2].mlp).__name__}"
    )


def test_legacy_no_moe_indices_replaces_all_layers(tmp_path, monkeypatch):
    """Backwards-compat: when moe_layer_indices is absent, every layer is MoE."""
    from outlier_engine.paging import _HybridPagedMLP, load_hybrid_paged_qwen

    cfg = _tiny_v32_config()
    cfg.pop("moe_layer_indices")
    cfg.pop("v3_2_moe_layers")
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    import outlier_engine.paging as paging_mod

    class _StubPageManager:
        def __init__(self, *a, **kw):
            self.roe_top_k = 0
        def cache_stats(self): return {}
        def prefetch_stats(self): return {}
        def et_routing_stats(self): return {}
        def enable_expert_prefetch(self): pass
        def enable_et_routing(self): pass
        def enable_roe(self, *_): pass

    monkeypatch.setattr(paging_mod, "ExpertPageManager", _StubPageManager)

    model = load_hybrid_paged_qwen(str(tmp_path), device="cpu")
    for idx, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp, _HybridPagedMLP), (
            f"Legacy path should replace every layer; layer {idx} is {type(layer.mlp).__name__}"
        )
