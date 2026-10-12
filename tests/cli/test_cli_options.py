from studio.cli.main import build_parser


def test_repeated_only_and_skip_add_up() -> None:
    args = build_parser().parse_args(["run", "x", "--only", "roughcut", "--only", "reels", "export",
                                 "--skip", "text", "--skip", "program"])
    assert args.only == ["roughcut", "reels", "export"]
    assert args.skip == ["text", "program"]
