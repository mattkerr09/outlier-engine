from outlier_engine.cli import build_parser


def test_cli_parses_run_command():
    parser = build_parser()
    args = parser.parse_args(["run", "Outlier-Ai/Outlier-10B", "Hello", "--max-tokens", "10"])

    assert args.command == "run"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.prompt == "Hello"
    assert args.max_tokens == 10


def test_bench_command_parses_args():
    parser = build_parser()
    args = parser.parse_args(["bench", "Outlier-Ai/Outlier-10B", "--device", "mps"])

    assert args.command == "bench"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.device == "mps"
    assert args.max_tokens == 20
