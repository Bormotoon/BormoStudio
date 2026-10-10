"""Synchronization planning contracts and legacy-compatible complex placement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from studio.core.project import Asset, AssetRole, Project, stable_fingerprint
from studio.core.timeline import AudioWarpMap, AudioWarpPiece, SourcePlacement, TimeDomain
from studio.core.transcript import Transcript
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.sync_alignment import align_sources
from studio.stages.sync_engine import render_aligned_clip
from studio.stages.sync_geometry import PieceConfig


@dataclass(frozen=True, slots=True)
class TextAnchor:
    camera_s: float
    recorder_s: float
    token: str


class SyncStage:
    id = "sync"
    title = "Synchronization"
    after: tuple[str, ...] = ("transcribe",)
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return [Requirement("binary", "ffmpeg", "render synchronized audio")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        cameras = [asset for asset in project.assets if asset.role is AssetRole.CAMERA]
        if not cameras:
            return Decision.skip("no camera assets")
        if len(cameras) == 1 and not any(
            asset.role is AssetRole.RECORDER for asset in project.assets
        ):
            return Decision.skip("single camera needs no synchronization")
        required = _required_transcripts(project)
        missing = [asset_id for asset_id in required if asset_id not in project.transcripts]
        if missing:
            return Decision.blocked(
                f"missing transcripts: {', '.join(missing)}", "run transcription"
            )
        return Decision.run({"mode": _sync_mode(settings)})

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        transcripts = {
            asset_id: path.read_text(encoding="utf-8")
            for asset_id, path in project.transcripts.items()
            if asset_id in {asset.id for asset in project.assets} and path.is_file()
        }
        from studio.stages.transcribe import _sha256

        return stable_fingerprint(
            "sync-v4", transcripts, project.assets,
            {asset.id: _sha256(project.source_dir / asset.path)
             for asset in project.assets if asset.role in {AssetRole.CAMERA, AssetRole.RECORDER}},
            settings.get("sync", {}), settings.get("transcribe", {}).get("audio_streams", {}),
        )

    def run(self, context: StageContext) -> StageOutput:
        mode = _sync_mode(context.settings)
        if mode == "camera" or not any(
            asset.role is AssetRole.RECORDER for asset in context.project.assets
        ):
            return _run_camera_placement(context)
        if mode == "complex" or (mode in {"auto", "simple"} and any(
            asset.role is AssetRole.RECORDER for asset in context.project.assets
        )):
            return _run_complex(context)
        cameras = [asset for asset in context.project.assets if asset.role is AssetRole.CAMERA]
        recorders = [asset for asset in context.project.assets if asset.role is AssetRole.RECORDER]
        placements: list[SourcePlacement] = []
        warps: list[AudioWarpMap] = []
        for index, camera in enumerate(cameras):
            duration = _duration(camera)
            camera_offset = 0.0
            provenance = "metadata"
            if index and not recorders:
                primary = Transcript.load(context.project.transcripts[cameras[0].id])
                candidate = Transcript.load(context.project.transcripts[camera.id])
                anchors = text_anchors(primary, candidate)
                camera_offset, _camera_k, _residual = fit_alignment(anchors)
                provenance = "text"
            placements.append(
                SourcePlacement(
                    camera.id,
                    camera_offset,
                    0.0,
                    duration,
                    provenance=provenance,
                )
            )
        if recorders:
            reference = recorders[0]
            recorder_transcript = Transcript.load(context.project.transcripts[reference.id])
            for camera in cameras:
                camera_transcript = Transcript.load(context.project.transcripts[camera.id])
                anchors = text_anchors(camera_transcript, recorder_transcript)
                offset, k, residual = fit_alignment(anchors)
                duration = _duration(reference)
                placements.append(
                    SourcePlacement(
                        reference.id,
                        offset,
                        0.0,
                        duration,
                        k,
                        "text",
                        {"inliers": float(len(anchors)), "residual_ms": residual * 1000},
                    )
                )
                selected_mode = choose_sync_mode(mode, duration, k, residual)
                strategy = 3 if selected_mode == "complex" else 1
                warps.append(
                    AudioWarpMap(
                        f"warp-{reference.id}-{camera.id}",
                        reference.id,
                        camera.id,
                        (AudioWarpPiece(0.0, duration, 0.0, duration * k, "resample"),),
                        TimeDomain.FILE,
                        strategy,
                        {"anchors": float(len(anchors)), "residual_ms": residual * 1000},
                    )
                )
        output = context.work_dir / "stages" / "sync" / "output.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "mode": mode,
            "placements": [
                {
                    "asset_id": item.asset_id,
                    "offset_s": item.offset_s,
                    "in_s": item.in_s,
                    "duration_s": item.duration_s,
                    "k": item.k,
                    "provenance": item.provenance,
                    "evidence": item.evidence,
                }
                for item in placements
            ],
            "audio_warp_maps": [warp.id for warp in warps],
        }
        output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return StageOutput(
            (output,), {"placements": placements, "audio_warp_maps": warps}
        )


def text_anchors(camera: Transcript, recorder: Transcript) -> list[TextAnchor]:
    camera_words: dict[str, list[float]] = {}
    recorder_words: dict[str, list[float]] = {}
    for word in camera.words:
        camera_words.setdefault(_normalize(word.text), []).append(word.start)
    for word in recorder.words:
        recorder_words.setdefault(_normalize(word.text), []).append(word.start)
    return [
        TextAnchor(camera_words[token][0], recorder_words[token][0], token)
        for token in camera_words.keys() & recorder_words.keys()
        if token and len(camera_words[token]) == len(recorder_words[token]) == 1
    ]


def _run_complex(context: StageContext) -> StageOutput:
    from studio.stages.sync_media import selected_media

    context.work_dir.mkdir(parents=True, exist_ok=True)
    with selected_media(context.project, context.work_dir,
                        context.settings.get("transcribe", {}).get("audio_streams")) as project:
        return _render_complex(StageContext(project, context.settings, context.work_dir))


def _render_complex(context: StageContext) -> StageOutput:
    project = context.project
    cameras = _ordered_cameras(project, context.settings)
    recorders = [asset for asset in project.assets if asset.role is AssetRole.RECORDER]
    if not recorders:
        raise ValueError("recorder rendering requires a recorder")
    if not cameras:
        raise ValueError("complex rendering requires a camera")
    artifacts = []
    warps = []
    outputs = {key: paths for key, paths in project.outputs.items()
               if not key.startswith(("sync:", "sync-tracks:", "voice-segments:", "ambience:"))
               and key not in {"master_wav", "self_check", "repair", "verify", "audio_processing"}}
    sync = context.settings.get("sync", {})
    sync = sync if isinstance(sync, dict) else {}
    if sync.get("self_check") == "repair" and sync.get("voice_enhance", "off") != "off":
        raise ValueError("sync.self_check=repair currently requires voice_enhance=off")
    mode = _sync_mode(context.settings)
    from studio.stages.sync_sessions import plan_sessions

    transcripts = {asset.id: Transcript.load(project.transcripts[asset.id])
                   for asset in [*recorders, *cameras]}
    camera_transcripts = [transcripts[camera.id] for camera in cameras]
    placements, fits, rejected = plan_sessions(
        project, cameras, recorders, transcripts, sync, acoustic_first=mode in {"auto", "simple"},
    )
    repair_sources = {}
    for camera, candidate in zip(cameras, camera_transcripts, strict=True):
        available = [asset for asset in recorders if (camera.id, asset.id) in fits]
        recorder = available[0]
        transcript = transcripts[recorder.id]
        source = project.source_dir / recorder.path
        alignment = fits[camera.id, recorder.id]
        voice = context.work_dir / "stages" / "sync" / "voice" / f"{camera.id}.wav"
        selected = choose_sync_mode(
            mode, candidate.duration, alignment.k, alignment.residual_ms / 1000,
            max_drift_ms=float(sync.get("max_drift_ms", 20.0)),
        )
        strategy = int(sync.get("strategy", 3)) if selected == "complex" else 1
        plan = render_aligned_clip(
            alignment, PieceConfig(), source, voice,
            clip_duration_s=candidate.duration, recorder_duration_s=transcript.duration,
            recorder_words=[(word.start, word.end) for word in transcript.words],
            strategy=strategy,
            source_asset_id=recorder.id, target_asset_id=camera.id,
            channels=int(recorder.manual.get("media_info", {}).get("audio_channels") or 1),
            camera_audio=project.source_dir / camera.path if sync.get("boundary_flex") else None,
        )
        warps.append(plan.warp)
        artifacts.append(voice)
        outputs[f"sync:{camera.id}"] = [voice]
        choices = [(recorder.id, voice, alignment.residual_ms, alignment.inliers)]
        for extra in available[1:]:
            extra_transcript = transcripts[extra.id]
            fitted = fits[camera.id, extra.id]
            extra_voice = voice.with_name(f"{camera.id}-{extra.id}.wav")
            extra_mode = choose_sync_mode(
                mode, candidate.duration, fitted.k, fitted.residual_ms / 1000,
                max_drift_ms=float(sync.get("max_drift_ms", 20.0)),
            )
            extra_plan = render_aligned_clip(
                fitted, PieceConfig(), project.source_dir / extra.path, extra_voice,
                clip_duration_s=candidate.duration, recorder_duration_s=extra_transcript.duration,
                recorder_words=[(word.start, word.end) for word in extra_transcript.words],
                strategy=int(sync.get("strategy", 3)) if extra_mode == "complex" else 1,
                source_asset_id=extra.id, target_asset_id=camera.id,
                channels=int(extra.manual.get("media_info", {}).get("audio_channels") or 1),
                camera_audio=(project.source_dir / camera.path
                              if sync.get("boundary_flex") else None),
            )
            warps.append(extra_plan.warp)
            artifacts.append(extra_voice)
            choices.append((extra.id, extra_voice, fitted.residual_ms, fitted.inliers))
        def choice_rank(
            item: tuple[str, Any, float, int], camera_id: str = camera.id,
            duration: float = candidate.duration,
        ) -> tuple[float, float, int]:
            fitted = fits[camera_id, item[0]]
            coverage = max(0.0, min(duration,
                fitted.offset + fitted.k * transcripts[item[0]].duration)
                - max(0.0, fitted.offset))
            return (-coverage, item[2], -item[3])

        choices.sort(key=choice_rank)
        outputs[f"sync:{camera.id}"] = [choices[0][1]]
        if sync.get("self_check") == "repair":
            selected_asset = next(asset for asset in recorders if asset.id == choices[0][0])
            repair_sources[camera.id] = (
                selected_asset.id, project.source_dir / selected_asset.path,
                Transcript.load(project.transcripts[selected_asset.id]),
                int(selected_asset.manual.get("media_info", {}).get("audio_channels") or 1),
            )
        if sync.get("recorder_mode") == "all":
            outputs[f"sync-tracks:{camera.id}"] = [item[1] for item in choices]
    report = context.work_dir / "stages" / "sync" / "output.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "schema_version": 1, "mode": mode,
        "strategies": {warp.target_asset_id: warp.strategy for warp in warps},
        "voices": {camera.id: str(outputs[f"sync:{camera.id}"][0].relative_to(report.parent))
                   for camera in cameras},
        "padding": "silence is outside invertible source maps",
        "rejected_pairs": rejected,
    }, indent=2) + "\n", encoding="utf-8")
    artifacts.append(report)
    if sync.get("voice_enhance", "off") != "off" or sync.get("ambience"):
        from studio.stages.sync_audio_steps import process_audio

        enhanced, ambience, processed = process_audio(
            {camera.id: outputs[f"sync:{camera.id}"][0] for camera in cameras},
            {camera.id: project.source_dir / camera.path for camera in cameras},
            report.parent / "audio-processing", sync,
        )
        artifacts.extend(processed)
        for asset_id, path in enhanced.items():
            outputs[f"sync:{asset_id}"] = [path]
        for asset_id, path in ambience.items():
            outputs[f"ambience:{asset_id}"] = [path]
        outputs["audio_processing"] = [processed[-1]]
    if sync.get("self_check", "off") != "off":
        from studio.stages.sync_check_step import check_voices

        checks = check_voices(
            dict(zip([camera.id for camera in cameras], camera_transcripts, strict=True)),
            {camera.id: outputs[f"sync:{camera.id}"][0] for camera in cameras},
            report.parent / "self-check", context.settings.get("transcribe", {}),
        )
        artifacts.extend(checks)
        outputs["self_check"] = list(checks)
        if sync.get("self_check") == "repair":
            from studio.stages.sync_repair import repair_voices

            repaired, repaired_maps, repair_artifacts = repair_voices(
                dict(zip([camera.id for camera in cameras], camera_transcripts, strict=True)),
                {camera.id: project.source_dir / camera.path for camera in cameras},
                repair_sources, report.parent / "repair", checks[-1],
                context.settings.get("transcribe", {}),
            )
            artifacts.extend(repair_artifacts)
            outputs["repair"] = list(repair_artifacts)
            for asset_id, path in repaired.items():
                old_path = outputs[f"sync:{asset_id}"][0]
                outputs[f"sync:{asset_id}"] = [path]
                lane_key = f"sync-tracks:{asset_id}"
                if lane_key in outputs:
                    outputs[lane_key] = [path if item == old_path else item
                                         for item in outputs[lane_key]]
                replacement = repaired_maps[asset_id]
                warps = [replacement if warp.id == replacement.id else warp for warp in warps]
    from studio.stages.sync_outputs import render_voice_master, segment_voice

    if float(sync.get("voice_segment_minutes", 0)) > 0:
        for camera in cameras:
            segments = segment_voice(
                outputs[f"sync:{camera.id}"][0], report.parent / "segments" / camera.id,
                float(sync["voice_segment_minutes"]),
            )
            artifacts.extend(segments)
            outputs[f"voice-segments:{camera.id}"] = list(segments)
    if sync.get("master_wav"):
        master = render_voice_master(
            placements, {camera.id: outputs[f"sync:{camera.id}"][0] for camera in cameras},
            context.work_dir / "export" / "voice-master.wav",
            crossfade_ms=float(sync.get("master_crossfade_ms", 10.0)),
        )
        artifacts.append(master)
        outputs["master_wav"] = [master]
    if sync.get("verify"):
        from studio.stages.sync_verify import measure

        measurements = []
        for camera, candidate in zip(cameras, camera_transcripts, strict=True):
            verification = measure(project.source_dir / camera.path,
                                   outputs[f"sync:{camera.id}"][0],
                                   source_duration_s=candidate.duration)
            status, reason = verification.verdict(20.0)
            measurements.append({"asset_id": camera.id, "status": status, "reason": reason,
                                 **verification.summary()})
        verify_path = report.with_name("verify.json")
        verify_path.write_text(json.dumps({"schema_version": 1, "clips": measurements},
                                          indent=2) + "\n", encoding="utf-8")
        artifacts.append(verify_path)
        outputs["verify"] = [verify_path]
    report_data = json.loads(report.read_text(encoding="utf-8"))
    report_data["voices"] = {
        camera.id: str(outputs[f"sync:{camera.id}"][0].relative_to(report.parent))
        for camera in cameras
    }
    report_data["strategies"] = {warp.id: warp.strategy for warp in warps}
    report.write_text(json.dumps(report_data, indent=2) + "\n", encoding="utf-8")
    return StageOutput(tuple(artifacts), {
        "placements": placements, "audio_warp_maps": warps, "outputs": outputs,
    })


def _run_camera_placement(context: StageContext) -> StageOutput:
    from studio.stages.sync_media import selected_media

    context.work_dir.mkdir(parents=True, exist_ok=True)
    with selected_media(context.project, context.work_dir,
                        context.settings.get("transcribe", {}).get("audio_streams")) as project:
        return _place_cameras(StageContext(project, context.settings, context.work_dir))


def _place_cameras(context: StageContext) -> StageOutput:
    project = context.project
    cameras = _ordered_cameras(project, context.settings)
    if not cameras:
        raise ValueError("no camera assets")
    reference = cameras[0]
    conf = context.settings.get("sync", {})
    conf = conf if isinstance(conf, dict) else {}
    primary = (Transcript.load(project.transcripts[reference.id])
               if reference.id in project.transcripts else
               Transcript(project.source_dir / reference.path, "", _duration(reference), []))
    offsets = conf.get("session_offsets", {})
    reference_offset = float(offsets.get(reference.id, offsets.get(reference.group_id, 0)))
    placements = [SourcePlacement(reference.id, reference_offset, 0, primary.duration,
                                  provenance="metadata")]
    for camera in cameras[1:]:
        candidate = (Transcript.load(project.transcripts[camera.id])
                     if camera.id in project.transcripts else
                      Transcript(project.source_dir / camera.path, "", _duration(camera), []))
        manual_offset = offsets.get(camera.id, offsets.get(camera.group_id))
        if manual_offset is not None:
            placements.append(SourcePlacement(camera.id, float(manual_offset), 0,
                                              candidate.duration, provenance="manual"))
            continue
        fitted = align_sources(
            primary, candidate, project.source_dir / reference.path,
            project.source_dir / camera.path, conf, acoustic_first=True,
        )
        placements.append(SourcePlacement(
            camera.id, reference_offset + fitted.offset, 0, candidate.duration,
            fitted.k, fitted.provenance,
            {"inliers": float(fitted.inliers), "residual_ms": fitted.residual_ms},
        ))
    origin = min(item.offset_s for item in placements)
    from dataclasses import replace

    placements = [replace(item, offset_s=item.offset_s - origin) for item in placements]
    output = context.work_dir / "stages" / "sync" / "output.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema_version": 1, "mode": "camera",
                                  "placement_ids": [item.asset_id for item in placements]},
                                 indent=2) + "\n", encoding="utf-8")
    return StageOutput((output,), {"placements": placements, "audio_warp_maps": [],
                                  "outputs": {key: paths for key, paths in project.outputs.items()
                                              if not key.startswith("sync:")}})


def fit_alignment(anchors: list[TextAnchor]) -> tuple[float, float, float]:
    if len(anchors) < 2:
        raise ValueError("at least two unique text anchors are required")
    xs = [anchor.recorder_s for anchor in anchors]
    ys = [anchor.camera_s for anchor in anchors]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0:
        raise ValueError("text anchors do not span recorder time")
    k = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    offset = mean_y - k * mean_x
    residual = (
        sum((y - (offset + k * x)) ** 2 for x, y in zip(xs, ys, strict=True)) / len(xs)
    ) ** 0.5
    return offset, k, residual


def choose_sync_mode(
    requested: str,
    duration_s: float,
    k: float,
    residual_s: float,
    *,
    max_drift_ms: float = 20.0,
) -> str:
    if requested != "auto":
        return requested
    accumulated_drift_ms = abs(k - 1.0) * duration_s * 1000
    if accumulated_drift_ms < max_drift_ms and residual_s < 0.02:
        return "simple"
    return "complex"


def _required_transcripts(project: Project) -> list[str]:
    path = project.work_dir / "stages" / "prepare" / "requirements.json"
    if not path.is_file():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("transcripts", []))


def _sync_mode(settings: dict[str, Any]) -> str:
    sync = settings.get("sync", {})
    return str(sync.get("mode", "auto")) if isinstance(sync, dict) else "auto"


def _duration(asset: Asset) -> float:
    return float(asset.manual.get("media_info", {}).get("duration_s", 0.0))


def _ordered_cameras(project: Project, settings: dict[str, Any]) -> list[Asset]:
    cameras = [asset for asset in project.assets if asset.role is AssetRole.CAMERA]
    selected = settings.get("sync", {}).get("reference_camera", "auto")
    if selected == "auto":
        return cameras
    matches = [asset for asset in cameras if asset.id == selected or asset.group_id == selected]
    if not matches:
        raise ValueError(f"unknown sync.reference_camera: {selected}")
    return [*matches, *(asset for asset in cameras if asset not in matches)]


def _normalize(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())
