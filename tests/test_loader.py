from __future__ import annotations

import json
from pathlib import Path

import torch

from outlier_engine.loader import inspect_model


def _make_config_only_model(tmp_path: Path) -> Path:
    cfg = {
        "n_layers": 2,
        "hidden_dim": 32,
        "n_heads": 2,
        "intermediate_dim": 64,
        "vocab_size": 128,
        "max_seq_len": 64,
        "outlier_num_experts": 4,
        "outlier_num_experts_per_tok": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    torch.save({"router": torch.randn(4, 32)}, tmp_path / "routers.pt")
    layer_dir = tmp_path / "layer_0"
    layer_dir.mkdir()
    torch.save({"dummy": torch.ones(1)}, layer_dir / "expert_0.pt")
    return tmp_path


def test_inspect_model_reads_config_and_artifacts(tmp_path):
    model_dir = _make_config_only_model(tmp_path)
    info = inspect_model(str(model_dir))

    assert info["config"]["n_layers"] == 2
    assert info["config"]["n_experts"] == 4
    assert info["artifacts"]["expert_pt_count"] == 1
