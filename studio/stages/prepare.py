"""Finalize the complete input set before compute planning."""

from __future__ import annotations

import json
from typing import Any

from studio.core.project import Asset, AssetRole, Project, stable_fingerprint
from studio.stages.base import (
    Decision,
    GpuUse,
    Requirement,
    StageContext,
    StageOutput,
    select_settings,
)
from studio.stages.locate import load_positions


class PrepareStage:
    id = "prepare"
    title = "Prepare inputs"
    after: tuple[str, ...] = ("locate",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = (
        "sync.mode", "sync.self_check", "transcribe.audio_streams", "transcribe.recorder_scope",
        "transcribe.ranges", "transcribe.window_margin_s", "transcribe.camera_transcripts",
        "transcribe.camera_mode", "sync.placement",
    )
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return []

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        return Decision.run() if project.assets else Decision.blocked(
            "scan found no media", "choose a folder containing supported media"
        )

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        inputs = [(asset.id, asset.role.value, asset.manual) for asset in project.assets]
        positions = project.work_dir / "stages" / "locate" / "positions.json"
        return stable_fingerprint(
            "prepare-v5", inputs, project.plan_revision,
            select_settings(settings, self.settings_keys),
            positions.read_text(encoding="utf-8") if positions.is_file() else None,
        )

    def run(self, context: StageContext) -> StageOutput:
        project = context.project
        recorders = [asset for asset in project.assets if asset.role is AssetRole.RECORDER]
        cameras = [asset for asset in project.assets if asset.role is AssetRole.CAMERA]
        sync = context.settings.get("sync", {})
        transcribe = context.settings.get("transcribe", {})
        mode = str(sync.get("mode", "auto"))
        positions = load_positions(project).get("positions", {})
        located = {camera.id for camera in cameras if positions.get(camera.id)}
        transcripts = [] if mode == "camera" else [asset.id for asset in recorders]
        reasons: dict[str, str] = {
            asset_id: "primary recorder transcript" for asset_id in transcripts
        }
        text_first = mode != "camera" and bool(recorders) and (
            sync.get("placement", "text") == "text")
        always = transcribe.get("camera_transcripts", "auto") == "always"
        self_check = sync.get("self_check", "off") != "off"
        camera_mode = str(transcribe.get("camera_mode", "fast"))
        modes: dict[str, str] = {}
        for camera in cameras:
            why = ("camera mode places cameras by their speech" if mode == "camera"
                   else "no recorder" if not recorders
                   else "sync.placement=text places every camera by its speech"
                   if text_first
                   else "transcribe.camera_transcripts=always" if always
                   else "self-check compares the rendered voice with the camera" if self_check
                   else "not located acoustically; text placement needs it"
                   if camera.id not in located else None)
            if why is not None:
                transcripts.append(camera.id)
                reasons[camera.id] = why
                # Only placement reads these words; a camera without a recorder
                # is the content itself and keeps transcribe.mode.
                if recorders and mode != "camera":
                    modes[camera.id] = camera_mode
        ranges: dict[str, list[list[float]]]
        if text_first:
            # Text placement searches each camera in the whole recorder.
            ranges, range_reasons = {}, {
                asset.id: "whole recorder: sync.placement=text" for asset in recorders}
        else:
            ranges, range_reasons = transcribe_ranges(
                recorders, cameras, positions,
                transcribe if isinstance(transcribe, dict) else {},
            ) if mode != "camera" else ({}, {})
        reasons.update(range_reasons)
        # Words are matched between transcripts of one Whisper mode: on
        # sony_run the fast recorder transcript gave anchors with p90 error
        # 38-58 ms against 117-129 ms for the quality one (worse timestamps),
        # and half the false anchors. The content keeps transcribe.mode.
        match_transcripts = {
            asset.id: camera_mode for asset in recorders
            if text_first and camera_mode != str(transcribe.get("mode", "fast"))}
        for asset_id in match_transcripts:
            reasons[f"match:{asset_id}"] = (
                f"{camera_mode} transcript of the recorder for matching camera words")
        data = {
            "schema_version": 2,
            "plan_revision": project.plan_revision,
            "assets": [asset.id for asset in cameras + recorders],
            "transcripts": list(dict.fromkeys(transcripts)),
            "transcribe_ranges": ranges,
            "transcribe_modes": modes,
            "match_transcripts": match_transcripts,
            "audio_streams": {
                asset.id: context.settings.get("transcribe", {}).get("audio_streams", {}).get(
                    asset.id, asset.manual.get("media_info", {}).get("audio_stream_index")
                )
                for asset in project.assets
                if asset.role in {AssetRole.CAMERA, AssetRole.RECORDER}
            },
            "modules": [],
            "reasons": reasons,
        }
        output = context.work_dir / "stages" / "prepare" / "requirements.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return StageOutput((output,))


# A recorder whose camera windows cover this much of it is transcribed whole.
WHOLE_FILE_FRACTION = 0.8


def transcribe_ranges(
    recorders: list[Asset], cameras: list[Asset], positions: dict[str, Any],
    transcribe: dict[str, Any],
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """Recorder ranges that need a transcript; a recorder absent from the result is whole."""
    scope = str(transcribe.get("recorder_scope", "auto"))
    margin = float(transcribe.get("window_margin_s", 60.0))
    manual = transcribe.get("ranges", {}) or {}
    result: dict[str, list[list[float]]] = {}
    reasons: dict[str, str] = {}
    unlocated = [camera.id for camera in cameras if not positions.get(camera.id)]
    for recorder in recorders:
        duration = float(recorder.manual.get("media_info", {}).get("duration_s") or 0.0)
        if recorder.id in manual:
            result[recorder.id] = [[float(a), float(b)] for a, b in manual[recorder.id]]
            reasons[recorder.id] = "transcribe.ranges"
            continue
        if scope == "full":
            reasons[recorder.id] = "whole recorder: transcribe.recorder_scope=full"
            continue
        if scope == "auto" and unlocated:
            reasons[recorder.id] = (
                f"whole recorder: camera(s) {', '.join(unlocated)} not located")
            continue
        spans = sorted(
            (max(0.0, entry["position_s"] - margin),
             min(duration, entry["position_s"] + _duration(camera) + margin))
            for camera in cameras
            for recorder_id, entry in positions.get(camera.id, {}).items()
            if recorder_id == recorder.id
        )
        merged: list[list[float]] = []
        for start, end in spans:
            if end <= start:
                continue
            if merged and start - merged[-1][1] < 2 * margin:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        covered = sum(end - start for start, end in merged)
        if not merged:
            reasons[recorder.id] = "whole recorder: no camera located in it"
        elif scope == "auto" and duration and covered >= WHOLE_FILE_FRACTION * duration:
            reasons[recorder.id] = (
                f"whole recorder: cameras cover {covered / duration:.0%} of it")
        else:
            result[recorder.id] = [[round(start, 3), round(end, 3)] for start, end in merged]
            reasons[recorder.id] = (
                f"recorder around cameras: {covered / 60:.1f} of {duration / 60:.1f} min")
    return result, reasons


def _duration(asset: Asset) -> float:
    return float(asset.manual.get("media_info", {}).get("duration_s") or 0.0)
