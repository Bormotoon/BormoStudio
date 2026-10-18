"""Tests use the same ffmpeg the application would choose (libass, NVENC)."""

from studio.core.ffmpeg_select import choose_ffmpeg


def pytest_configure() -> None:
    choose_ffmpeg()
