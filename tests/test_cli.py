import pytest

from outlier_engine import cli
from outlier_engine.cli import build_parser


def test_cli_parses_run_command():
    parser = build_parser()
    args = parser.parse_args(["run", "Outlier-Ai/Outlier-10B", "Hello", "--max-tokens", "10"])

    assert args.command == "run"
    assert args.inputs == ["Outlier-Ai/Outlier-10B", "Hello"]
    assert args.max_tokens == 10


def test_cli_parses_prompt_only_run_command():
    parser = build_parser()
    args = parser.parse_args(["run", "--paged", "Hello"])

    assert args.command == "run"
    assert args.inputs == ["Hello"]
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.paged is True


def test_bench_command_parses_args():
    parser = build_parser()
    args = parser.parse_args(["bench", "Outlier-Ai/Outlier-10B", "--device", "mps"])

    assert args.command == "bench"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.device == "mps"
    assert args.max_tokens == 20


def test_demo_command_parses_args():
    parser = build_parser()
    args = parser.parse_args(["demo", "--paged", "--max-tokens", "8"])

    assert args.command == "demo"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.paged is True
    assert args.max_tokens == 8


def test_repack_command_parses_args():
    parser = build_parser()
    args = parser.parse_args(["repack", "Outlier-Ai/Outlier-10B", "--output-dir", "/tmp/packed"])

    assert args.command == "repack"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.output_dir == "/tmp/packed"


def test_cli_run_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--help"])

    assert exc.value.code == 0
    assert "usage: outlier-engine run" in capsys.readouterr().out


def test_cli_demo_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["demo", "--help"])

    assert exc.value.code == 0
    assert "usage: outlier-engine demo" in capsys.readouterr().out


def test_cli_bench_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["bench", "--help"])

    assert exc.value.code == 0
    assert "usage: outlier-engine bench" in capsys.readouterr().out


def test_cli_info_prints_model_name(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "inspect_model",
        lambda model, token=None: {
            "model_ref": model,
            "resolved_model_ref": model,
            "model_dir": "hf://mock",
            "config": {"n_experts": 8, "top_k": 2, "total_params_B": 10.6},
            "artifacts": {},
        },
    )

    rc = cli.main(["info", "Outlier-Ai/Outlier-10B"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Outlier-Ai/Outlier-10B" in out
    assert "params=10.6B" in out
