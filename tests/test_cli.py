from outlier_engine.cli import build_parser


def test_cli_parses_run_command():
    parser = build_parser()
    args = parser.parse_args(["run", "Outlier-Ai/Outlier-10B", "Hello", "--max-tokens", "10"])

    assert args.command == "run"
    assert args.model == "Outlier-Ai/Outlier-10B"
    assert args.prompt == "Hello"
    assert args.max_tokens == 10
