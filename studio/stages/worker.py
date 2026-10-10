"""Subprocess entry point for one Studio stage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import yaml

from studio.core.digest import configure_from_settings
from studio.core.project import Project
from studio.stages.base import Stage, StageContext
from studio.stages.events import EventType, StageEvent
from studio.stages.export import ExportStage
from studio.stages.locate import LocateStage
from studio.stages.prepare import PrepareStage
from studio.stages.program import ProgramStage
from studio.stages.program_subtitles import ProgramSubtitlesStage
from studio.stages.reel_render import ReelRenderStage
from studio.stages.reels import ReelsStage
from studio.stages.roughcut import RoughcutStage
from studio.stages.scan import ScanStage
from studio.stages.speakers import SpeakersStage
from studio.stages.sync import SyncStage
from studio.stages.text import TextStage
from studio.stages.timeline import TimelineStage
from studio.stages.transcribe import TranscribeStage


def stage_registry() -> dict[str, Stage]:
    stages: list[Stage] = [
        ScanStage(),
        LocateStage(),
        PrepareStage(),
        TranscribeStage(),
        SyncStage(),
        TimelineStage(),
        SpeakersStage(),
        RoughcutStage(),
        TextStage(),
        ReelsStage(),
        ReelRenderStage(),
        ExportStage(),
        ProgramStage(),
        ProgramSubtitlesStage(),
    ]
    return {stage.id: stage for stage in stages}


PROGRESS_INTERVAL_S = 10.0


def _emit(event: StageEvent) -> None:
    print(event.to_json(), flush=True)


def execute(
    stage_id: str,
    project_path: Path,
    settings_path: Path,
    output_dir: Path,
    result_project: Path,
) -> int:
    stage = stage_registry().get(stage_id)
    if stage is None:
        raise ValueError(f"unknown stage: {stage_id}")
    project = Project.load(project_path)
    settings: dict[str, Any] = {}
    if settings_path.is_file():
        loaded = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError("settings root must be a mapping")
            settings = loaded
    configure_from_settings(settings)
    _emit(StageEvent(EventType.START, stage_id))
    events = sys.stdout  # stage code prints go to stderr; events stay on stdout
    last = [0.0, -1.0]

    def progress(value: float, message: str) -> None:
        # At most one event per PROGRESS_INTERVAL_S: logs show life, not noise.
        now = time.monotonic()
        value = min(1.0, max(0.0, float(value)))
        if now - last[0] < PROGRESS_INTERVAL_S and value < 1.0:
            return
        last[0], last[1] = now, value
        print(StageEvent(EventType.PROGRESS, stage_id,
                         {"value": round(value, 4), "message": message}).to_json(),
              file=events, flush=True)

    with redirect_stdout(sys.stderr):
        output = stage.run(StageContext(project, settings, output_dir, progress))
    for key, value in output.project_changes.items():
        if not hasattr(project, key):
            raise ValueError(f"stage returned unknown project field: {key}")
        setattr(project, key, value)
    project.save(result_project)
    for message in output.warnings:
        _emit(StageEvent(EventType.WARNING, stage_id, {"message": message}))
    for artifact in output.artifacts:
        _emit(StageEvent(EventType.ARTIFACT, stage_id, {"path": str(artifact)}))
    _emit(StageEvent(EventType.DONE, stage_id, {"status": "ok", "output": str(result_project)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_id")
    parser.add_argument("project", type=Path)
    parser.add_argument("settings", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result-project", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(
            args.stage_id, args.project, args.settings, args.output_dir, args.result_project
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        _emit(StageEvent(EventType.DONE, args.stage_id, {"status": "failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
