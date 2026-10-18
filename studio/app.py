"""Studio application entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.resources import files

from studio import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bstudio-gui",
        description="Local synchronized podcast production workflow.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check-resources",
        action="store_true",
        help="verify resources required by packaged builds",
    )
    return parser


def packaged_resources() -> tuple[str, ...]:
    root = files("studio.resources")
    return (
        root.joinpath("prompts", "ru", "proofread_default.txt").read_text(encoding="utf-8"),
        root.joinpath("fonts", "MontserratBlack.ttf").read_bytes().hex(),
        root.joinpath("subtitle_editor", "subtitles.html").read_text(encoding="utf-8"),
        files("studio.gui").joinpath("theme.qss").read_text(encoding="utf-8"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_resources:
        resources = packaged_resources()
        if not all(resources):
            raise RuntimeError("one or more packaged resources are empty")
        print("Studio package resources: OK")
    return 0


def gui_main() -> int:
    from studio.core.ffmpeg_select import choose_ffmpeg
    from studio.gui.main_window import main as launch_gui

    choose_ffmpeg()

    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
