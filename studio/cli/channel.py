"""Bounded channel discovery and resumable project acquisition."""

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from studio.core.workspace import project_lock, published
from studio.modules.cancellation import InstallationCancelled, check_cancelled, run_cancellable
from studio.stages.fetch import _yt_dlp_command


def discover_channel(
    url: str, limit: int, cancelled: Callable[[], bool] = lambda: False,
    filters: dict[str, Any] | None = None,
) -> list[str]:
    if not 1 <= limit <= 1000:
        raise ValueError("channel limit must be between 1 and 1000")
    filters = filters or {}
    filtered = bool(filters.get("exclude") or filters.get("skip_live", False)
                    or filters.get("min_duration_s") or filters.get("max_duration_s"))
    # Filters apply before the limit, so look further when they can drop videos.
    depth = min(1000, limit * 5) if filtered else limit
    result = run_cancellable(
        [
            *_yt_dlp_command(),
            "--flat-playlist",
            "--dump-single-json",
            "--playlist-end",
            str(depth),
            "--",
            url,
        ],
        cancelled,
        timeout=120,
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout,
                                            result.stderr)
    payload = json.loads(result.stdout)
    ids = []
    for entry in payload.get("entries", []):
        if isinstance(entry, dict) and re.fullmatch(r"[A-Za-z0-9_-]{11}", str(entry.get("id", ""))):
            if entry["id"] not in ids and _wanted(entry, filters):
                ids.append(entry["id"])
    return ids[:limit]


def _wanted(entry: dict[str, Any], filters: dict[str, Any]) -> bool:
    if entry["id"] in set(filters.get("exclude", [])):
        return False
    live = entry.get("live_status") in {"is_live", "is_upcoming", "post_live"}
    if filters.get("skip_live", False) and (live or entry.get("is_live")):
        return False
    # Zero or missing duration means unknown, not short: keep it.
    duration = entry.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        minimum = float(filters.get("min_duration_s", 0) or 0)
        maximum = float(filters.get("max_duration_s", 0) or 0)
        if (minimum and duration < minimum) or (maximum and duration > maximum):
            return False
    return True


def acquire_channel(
    url: str,
    destination: Path,
    limit: int,
    fetch: Callable[[str, Path, Callable[[], bool]], Any],
    cancelled: Callable[[], bool] = lambda: False,
    filters: dict[str, Any] | None = None,
) -> tuple[list[Path], list[dict[str, str]]]:
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    sources: list[Path] = []
    with project_lock(destination):
        for video_id in discover_channel(url, limit, cancelled, filters):
            check_cancelled(cancelled)
            folder = destination / video_id
            try:
                if (folder / "_studio" / "project.json").is_file():
                    from studio.core.project import Project

                    existing = Project.load(folder / "_studio" / "project.json")
                    if existing.source_dir.resolve() != folder.resolve():
                        raise ValueError("channel project belongs to another source directory")
                    status = "reused"
                else:
                    # Incomplete downloads remain intact for inspection; never overwrite.
                    fetch(f"https://www.youtube.com/watch?v={video_id}", folder, cancelled)
                    status = "fetched"
                sources.append(folder)
                results.append({"id": video_id, "status": status, "source": str(folder)})
            except InstallationCancelled:
                results.append({"id": video_id, "status": "cancelled", "source": str(folder)})
                write_report(destination / "channel-acquisition.json",
                             {"schema_version": 1, "videos": results})
                raise
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                results.append({"id": video_id, "status": "failed", "error": str(exc)})
            write_report(
                destination / "channel-acquisition.json", {"schema_version": 1, "videos": results}
            )
    return sources, results


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with published(path) as temporary:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
