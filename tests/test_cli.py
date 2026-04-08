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
