"""Fetch one YouTube URL into a Studio project without shell execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studio.core.project import Project, stable_fingerprint
from studio.modules.cancellation import InstallationCancelled, run_cancellable
from studio.modules.manager import ModuleManager
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput


class FetchStage:
    id = "fetch"
    title = "YouTube fetch"
    after: tuple[str, ...] = ()
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return [Requirement("module", "youtube", "download YouTube source media")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        fetch = settings.get("fetch", {})
        url = fetch.get("url") if isinstance(fetch, dict) else None
        return Decision.run({"url": url}) if url else Decision.skip("input is a local folder")

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        return stable_fingerprint("fetch-v1", settings.get("fetch", {}))

    def run(self, context: StageContext) -> StageOutput:
        fetch = context.settings.get("fetch", {})
        if not isinstance(fetch, dict) or not fetch.get("url"):
            raise ValueError("fetch.url is required")
        media, info = fetch_to_directory(
            str(fetch["url"]), context.work_dir / "fetch", options=yt_dlp_options(fetch)
        )
        return StageOutput((media, info))


def yt_dlp_options(conf: object) -> list[str]:
    """yt-dlp arguments for the `fetch` settings section."""
    conf = conf if isinstance(conf, dict) else {}
    options: list[str] = []
    height = int(conf.get("max_height", 0) or 0)
    if conf.get("download", "video") == "audio":
        options += ["--format", "bestaudio/best"]
    elif height:
        options += ["--format", f"bv*[height<={height}]+ba/b[height<={height}]/b"]
    if conf.get("cookies_file"):
        options += ["--cookies", str(conf["cookies_file"])]
    if "retries" in conf:
        options += ["--retries", str(int(conf["retries"]))]
    if conf.get("rate_limit"):
        options += ["--limit-rate", str(conf["rate_limit"])]
    return options


INCOMPLETE = ".fetch-incomplete"


def fetch_to_directory(
    url: str, destination: Path, cancelled: Callable[[], bool] = lambda: False,
    *, options: list[str] | None = None,
) -> tuple[Path, Path]:
    """Download one URL without executing a shell or overwriting local media.

    Cancellation stops yt-dlp mid-download and keeps its partial file; the
    marker records the URL so a later call for the same URL resumes it.
    """
    destination.mkdir(parents=True, exist_ok=True)
    (destination / INCOMPLETE).write_text(url + "\n", encoding="utf-8")
    output = destination / "source.%(ext)s"
    command = [
        *_yt_dlp_command(),
        "--no-playlist",
        "--no-overwrites",
        "--continue",
        "--write-info-json",
        *(options or []),
        "--output",
        str(output),
        "--",
        url,
    ]
    try:
        result = run_cancellable(command, cancelled)
    except InstallationCancelled:
        raise InstallationCancelled(
            "Download cancelled; the partial file is kept and the same command resumes it."
        ) from None
    if result.returncode:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    media = [
        path for path in destination.iterdir()
        if path.is_file() and not path.name.endswith((".info.json", ".part", ".ytdl"))
        and path.name != INCOMPLETE
    ]
    info = list(destination.glob("*.info.json"))
    if len(media) != 1 or len(info) != 1:
        raise RuntimeError("yt-dlp did not publish exactly one media and info file")
    (destination / INCOMPLETE).unlink()
    return media[0], info[0]


def resumable_fetch(destination: Path, url: str) -> bool:
    """Whether ``destination`` only holds an interrupted fetch of ``url``."""
    marker = destination / INCOMPLETE
    try:
        if marker.read_text(encoding="utf-8").strip() != url:
            return False
    except OSError:
        return False
    return all(
        path.is_file() and (path.name == INCOMPLETE or path.name.startswith("source."))
        for path in destination.iterdir()
    )


def _yt_dlp_command() -> list[str]:
    manager = ModuleManager()
    if manager.installed("youtube"):
        return [str(manager.interpreter("youtube")), "-m", "yt_dlp"]
    return ["yt-dlp"]


def read_fetch_metadata(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YouTube info JSON root must be an object")
    return value
