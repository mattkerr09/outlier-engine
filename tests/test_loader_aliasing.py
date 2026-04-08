from __future__ import annotations

import json

from outlier_engine.loader import load_model


class _DummyHFModel:
    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self


def test_paged_request_uses_canonical_alias_and_falls_back_for_dense_model(tmp_path, monkeypatch):
    cfg = {
        "model_type": "qwen2",
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 2,
        "intermediate_size": 64,
        "vocab_size": 128,
        "max_position_embeddings": 64,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    def fake_resolve(model_ref, token=None, *, canonicalize=True):
        assert model_ref == "Outlier-Ai/Outlier-10B"
        assert canonicalize is True
        return tmp_path

    monkeypatch.setattr("outlier_engine.loader._resolve_model_dir", fake_resolve)
    monkeypatch.setattr("outlier_engine.loader.load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: _DummyHFModel(),
    )

    loaded = load_model("Outlier-Ai/Outlier-10B", paged=True, device="cpu")

    assert loaded.model_ref == "Outlier-Ai/Outlier-10B-V2"
    assert loaded.paged is False
    assert loaded.backend == "hf"
