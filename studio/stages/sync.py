"""Synchronization planning contracts and legacy-compatible complex placement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studio.core.project import Asset, AssetRole, Project, stable_fingerprint
from studio.core.settings import fingerprint_view
from studio.core.timeline import AudioWarpMap, AudioWarpPiece, SourcePlacement, TimeDomain
from studio.core.transcript import Transcript
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.sync_alignment import align_sources
from studio.stages.sync_engine import render_aligned_clip
from studio.stages.sync_geometry import PieceConfig
from studio.stages.sync_refine import RefineSettings
from studio.stages.sync_render import PauseDuck


@dataclass(frozen=True, slots=True)
class TextAnchor:
    camera_s: float
    recorder_s: float
    token: str


class SyncStage:
    id = "sync"
    title = "Synchronization"
    after: tuple[str, ...] = ("transcribe",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("sync", "transcribe.audio_streams")
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
            if asset_id.removeprefix("match:") in {asset.id for asset in project.assets}
            and path.is_file()
        }
        from studio.stages.transcribe import _sha256

        return stable_fingerprint(
            "sync-v6", transcripts, project.assets,
            {asset.id: _sha256(project.source_dir / asset.path, project.work_dir)
             for asset in project.assets if asset.role in {AssetRole.CAMERA, AssetRole.RECORDER}},
            fingerprint_view("sync", settings.get("sync", {})),
            settings.get("transcribe", {}).get("audio_streams", {}),
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
    originals = {asset.id: asset for asset in context.project.assets}
    with selected_media(context.project, context.work_dir,
                        context.settings.get("transcribe", {}).get("audio_streams")) as project:
        return _render_complex(StageContext(project, context.settings, context.work_dir,
                                            context.progress), originals)


def _render_complex(context: StageContext,
                    originals: dict[str, Asset] | None = None) -> StageOutput:
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
               if not key.startswith(("sync:", "sync-tracks:", "voice-segments:", "ambience:",
                                      "recorder:", "voice:"))
               and key not in {"master_wav", "self_check", "repair", "verify", "audio_processing"}}
    sync = context.settings.get("sync", {})
    sync = sync if isinstance(sync, dict) else {}
    if sync.get("self_check") == "repair" and sync.get("voice_enhance", "off") != "off":
        raise ValueError("sync.self_check=repair currently requires voice_enhance=off")
    mode = _sync_mode(context.settings)
    placement = str(sync.get("placement", "text"))
    from studio.stages.sync_sessions import plan_sessions

    # A camera located acoustically needs no transcript: its length comes from probe.
    transcripts = {asset.id: _transcript_or_probe(project, asset)
                   for asset in [*recorders, *cameras]}
    from studio.stages.locate import load_positions

    positions = load_positions(project)
    located = positions.get("positions", {})
    # Pairs locate searched in full without a match: only text can place them.
    text_only = {
        (camera_id, recorder_id)
        for camera_id, missing in positions.get("missing", {}).items()
        for recorder_id in missing
    }
    if placement == "text":
        # Words first for every clip; the acoustics search only around
        # neighbours placed by text, in the second pass.
        text_only = {(camera.id, recorder.id) for camera in cameras for recorder in recorders}
    search_ranges = {
        (camera.id, recorder_id): (max(0.0, entry["position_s"] - 30.0),
                                   entry["position_s"] + transcripts[camera.id].duration + 30.0)
        for camera in cameras
        for recorder_id, entry in located.get(camera.id, {}).items()
    }
    camera_transcripts = [transcripts[camera.id] for camera in cameras]
    refine_reports: dict[str, Any] = {}
    stage_warnings: list[str] = []
    drift_reports: dict[str, Any] = {}
    constant_only: set[tuple[str, str]] = set()
    drift_checks: dict[tuple[str, str], Any] = {}
    # Clips given the recorder jumps without acoustics of their own: their jumps
    # are absorbed by the piece plan (strategy 3). A clip with its own acoustic
    # steps keeps strategy 1: the refinement finds each step to a millisecond
    # (a click or the fine grid), which sentence pieces cannot.
    stepped: set[tuple[str, str]] = set()

    def check(target: Asset, source: Asset, alignment: Any) -> Any:
        refine = refine_settings(sync, target)
        if target.role is not AssetRole.CAMERA or source.role is not AssetRole.RECORDER or (
                refine is None):
            return alignment
        from studio.stages.sync_drift import check_alignment

        try:
            checked = check_alignment(
                alignment, project.source_dir / target.path, project.source_dir / source.path,
                clip_duration_s=transcripts[target.id].duration,
                recorder_duration_s=transcripts[source.id].duration, settings=refine,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            drift_reports[f"{target.id}:{source.id}"] = {"used": "text", "reason": str(exc)}
            return alignment
        drift_reports[f"{target.id}:{source.id}"] = checked.report
        drift_checks[target.id, source.id] = checked
        if checked.constant_only:
            constant_only.add((target.id, source.id))
            if mode == "complex":
                stage_warnings.append(f"{Path(target.path).name}: {checked.report['complex']}")
        return checked.alignment

    # Progress: fitting the camera-recorder pairs is most of the work (0-60 %),
    # rendering a voice per camera next (60-90 %), the checks last.
    def fitting(done: int, total: int, name: str) -> None:
        context.progress(0.6 * done / max(1, total), f"fitting {done + 1}/{total}: {name}")

    from studio.stages.sync_jumps import apply_jumps, recorder_jumps, unseen_jumps
    from studio.stages.sync_rescue import quality_transcriber, second_pass

    second_report: dict[str, Any] = {}
    jumps: list[Any] = []
    rescue = second_pass(
        cameras=cameras, recorders=recorders, transcripts=transcripts,
        originals=originals or {asset.id: asset for asset in project.assets},
        source_dir=project.source_dir, settings=context.settings,
        output_dir=context.work_dir / "stages" / "sync" / "rescue",
        report=second_report, warnings=stage_warnings,
        progress=lambda message: context.progress(0.55, message),
        transcriber=quality_transcriber(context.settings, project.source_dir,
                                        project.work_dir),
        acoustic=placement == "text",
    )

    repairs: dict[str, Any] = {}

    def after_fitting(fit: Any, fits: dict[tuple[str, str], Any],
                      rejected: dict[str, str]) -> None:
        rescue(fit, fits, rejected)
        if sync.get("recorder_repair", "auto") != "off":
            # The recorder's own clock: samples it lost are found against
            # every camera and put back; every clip is refitted to the
            # repaired recorder by one line.
            from studio.core.parallel import cpu_workers
            from studio.stages.sync_recorder_clock import repair_recorders

            repairs.update(repair_recorders(
                source_dir=project.source_dir, cameras=cameras, recorders=recorders,
                originals=originals or {asset.id: asset for asset in project.assets},
                transcripts=transcripts, matching=matching, fits=fits, sync=sync,
                workers=cpu_workers(context.settings),
                output_dir=context.work_dir / "stages" / "sync" / "recorder",
                published_dir=project.work_dir / "stages" / "sync" / "recorder",
                scratch_dir=context.work_dir / "stages" / "sync" / "scratch",
                progress=lambda message: context.progress(0.58, message)))
            for recorder_id, repair in repairs.items():
                assets_by_id[recorder_id] = next(
                    asset for asset in recorders if asset.id == recorder_id)
                stage_warnings.append(
                    f"{Path(originals_by_id[recorder_id].path).name}: the recorder lost "
                    f"{repair.clock.total_ms / 1000:.1f} s in {len(repair.clock.applied)} "
                    "place(s); repaired"
                    + (" (filled from the cameras' sound)"
                       if sync.get("recorder_fill", "camera") == "camera" else ""))
        # One map of recorder jumps from every camera; clips without an
        # acoustic model of their own follow it (when the recorder was not repaired).
        jumps.extend(recorder_jumps(drift_checks, fits))
        for (camera_id, recorder_id), alignment in list(fits.items()):
            if camera_id not in transcripts or (camera_id, recorder_id) in stepped or (
                    recorder_id in repairs):
                continue
            checked = drift_checks.get((camera_id, recorder_id))
            if checked is not None and checked.model is not None:
                for jump in unseen_jumps(checked, jumps, transcripts[camera_id].duration):
                    stage_warnings.append(
                        f"{Path(assets_by_id[camera_id].path).name}: recorder jump of "
                        f"{jump.delta_ms:+.0f} ms at {jump.rec_s:.1f} s is not in its own "
                        "acoustic model; check by ear")
                continue
            bent = apply_jumps(alignment, [jump for jump in jumps
                                           if jump.recorder_id == recorder_id],
                               transcripts[camera_id].duration)
            if bent is not None:
                fits[camera_id, recorder_id] = bent
                stepped.add((camera_id, recorder_id))

    assets_by_id = {asset.id: asset for asset in [*cameras, *recorders]}
    originals_by_id = originals or assets_by_id
    matching = {asset.id: Transcript.load(project.transcripts[f"match:{asset.id}"])
                for asset in recorders
                if f"match:{asset.id}" in project.transcripts
                and project.transcripts[f"match:{asset.id}"].is_file()}
    placements, fits, rejected = plan_sessions(
        project, cameras, recorders, transcripts, sync,
        acoustic_first=placement == "acoustic" and mode in {"auto", "simple"},
        check=check, search_ranges=search_ranges, text_only=text_only, progress=fitting,
        skipped_search=("text placement searches acoustically between placed neighbours "
                        "in its second pass" if placement == "text" else None),
        second_pass=after_fitting, matching=matching,
    )
    camera_transcripts = [transcripts[camera.id] for camera in cameras]
    left_out = [camera for camera in cameras
                if not any(target == camera.id for target, _ in fits)]
    if left_out:
        stage_warnings.append(
            "left out, no recorder matches: "
            + ", ".join(Path(camera.path).name for camera in left_out))
        cameras = [camera for camera in cameras if camera not in left_out]
        camera_transcripts = [transcripts[camera.id] for camera in cameras]
    repair_sources = {}
    for number, (camera, candidate) in enumerate(zip(cameras, camera_transcripts, strict=True)):
        context.progress(0.6 + 0.3 * number / max(1, len(cameras)),
                         f"rendering voice {number + 1}/{len(cameras)}: {Path(camera.path).name}")
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
        if (camera.id, recorder.id) in constant_only:
            selected = "simple"  # the acoustics support only the constant plan
        if mode == "auto" and (camera.id, recorder.id) in stepped:
            selected = "complex"  # recorder jumps from other cameras, cut in pauses
        strategy = int(sync.get("strategy", 3)) if selected == "complex" else 1
        refine = refine_settings(sync, camera)
        plan = render_aligned_clip(
            alignment, piece_config(sync), source, voice,
            clip_duration_s=candidate.duration, recorder_duration_s=transcript.duration,
            recorder_words=[(word.start, word.end) for word in transcript.words],
            strategy=strategy,
            source_asset_id=recorder.id, target_asset_id=camera.id,
            channels=int(recorder.manual.get("media_info", {}).get("audio_channels") or 1),
            camera_audio=project.source_dir / camera.path if sync.get("boundary_flex") else None,
            refine_audio=project.source_dir / camera.path if refine else None,
            refine=refine, label=Path(camera.path).name,
            stretch_method=str(sync.get("stretch_method", "auto")),
            duck=PauseDuck.from_settings(
                sync, [(word.start, word.end) for word in candidate.words],
                [(word.start, word.end) for word in transcript.words]),
        )
        if plan.refine is not None:
            refine_reports[camera.id] = plan.refine
        stage_warnings.extend(plan.warnings)
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
            if (camera.id, extra.id) in constant_only:
                extra_mode = "simple"
            if mode == "auto" and (camera.id, extra.id) in stepped:
                extra_mode = "complex"
            extra_plan = render_aligned_clip(
                fitted, piece_config(sync), project.source_dir / extra.path, extra_voice,
                clip_duration_s=candidate.duration, recorder_duration_s=extra_transcript.duration,
                recorder_words=[(word.start, word.end) for word in extra_transcript.words],
                strategy=int(sync.get("strategy", 3)) if extra_mode == "complex" else 1,
                source_asset_id=extra.id, target_asset_id=camera.id,
                channels=int(extra.manual.get("media_info", {}).get("audio_channels") or 1),
                camera_audio=(project.source_dir / camera.path
                              if sync.get("boundary_flex") else None),
                refine_audio=project.source_dir / camera.path if refine else None,
                refine=refine, label=f"{Path(camera.path).name}/{Path(extra.path).name}",
                stretch_method=str(sync.get("stretch_method", "auto")),
                duck=PauseDuck.from_settings(
                    sync, [(word.start, word.end) for word in candidate.words],
                    [(word.start, word.end) for word in extra_transcript.words]),
            )
            if extra_plan.refine is not None:
                refine_reports[f"{camera.id}:{extra.id}"] = extra_plan.refine
            stage_warnings.extend(extra_plan.warnings)
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
    clip_report = {}
    for camera in cameras:
        pairs = [(recorder.id, fits[camera.id, recorder.id]) for recorder in recorders
                 if (camera.id, recorder.id) in fits]
        recorder_id, alignment = pairs[0]
        acoustic = alignment.provenance.startswith("acoustic")
        clip_report[camera.id] = {
            "recorder": recorder_id, "method": alignment.provenance,
            "precision": "acoustic" if acoustic else "text",
            "start_s": round(-alignment.offset / (alignment.k or 1.0), 3),
            "inliers": alignment.inliers, "residual_ms": round(alignment.residual_ms, 1),
            "notes": list(alignment.notes),
        }
        if not acoustic:
            stage_warnings.append(
                f"{Path(camera.path).name}: placed by its words only (about ±0.1 s); "
                "the acoustics did not confirm it")
    report = context.work_dir / "stages" / "sync" / "output.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "schema_version": 1, "mode": mode,
        "strategies": {warp.target_asset_id: warp.strategy for warp in warps},
        "voices": {camera.id: str(outputs[f"sync:{camera.id}"][0].relative_to(report.parent))
                   for camera in cameras},
        "padding": "silence is outside invertible source maps",
        "placement": placement,
        "clips": clip_report,
        "rejected_pairs": rejected,
        "second_pass": second_report,
        "recorder_jumps": [jump.to_dict() for jump in jumps],
        "recorder_clock": {recorder_id: repair.report() for recorder_id, repair in repairs.items()},
        "acoustic_refine": refine_reports,
        "acoustic_fit": drift_reports,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    context.progress(0.9, "checking rendered voices")
    if sync.get("self_check", "off") != "off":
        from studio.stages.sync_check_step import check_voices

        checks = check_voices(
            dict(zip([camera.id for camera in cameras], camera_transcripts, strict=True)),
            {camera.id: outputs[f"sync:{camera.id}"][0] for camera in cameras},
            report.parent / "self-check", self_check_transcribe(context.settings),
            cameras={camera.id: project.source_dir / camera.path for camera in cameras},
            sync_settings=sync,
        )
        artifacts.extend(checks)
        outputs["self_check"] = list(checks)
        if sync.get("self_check") == "repair":
            from studio.stages.sync_repair import repair_voices

            repaired, repaired_maps, repair_artifacts = repair_voices(
                dict(zip([camera.id for camera in cameras], camera_transcripts, strict=True)),
                {camera.id: project.source_dir / camera.path for camera in cameras},
                repair_sources, report.parent / "repair", checks[-1],
                self_check_transcribe(context.settings), sync,
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
    if sync.get("verify", True):
        from studio.stages.sync_verify import measure, short_clip_grid

        measurements = []
        threshold = float(sync.get("max_drift_ms", 20.0))
        for camera, candidate in zip(cameras, camera_transcripts, strict=True):
            grid_s, window_s = short_clip_grid(candidate.duration,
                                               float(sync.get("refine_grid_s", 2.5)),
                                               float(sync.get("refine_window_s", 4.0)))
            try:
                verification = measure(project.source_dir / camera.path,
                                       outputs[f"sync:{camera.id}"][0],
                                       grid_s=grid_s, window_s=window_s,
                                       source_duration_s=candidate.duration)
            except (RuntimeError, OSError, ValueError) as exc:
                # A camera without a readable audio stream cannot be measured.
                measurements.append({"asset_id": camera.id, "status": "inconclusive",
                                     "reason": f"camera audio could not be measured: {exc}"})
                continue
            status, reason = verification.verdict(threshold)
            measurements.append({"asset_id": camera.id, "status": status, "reason": reason,
                                 **verification.summary(threshold)})
        for item in measurements:
            if item["status"] != "passed":
                name = Path(next(camera.path for camera in cameras
                                 if camera.id == item["asset_id"])).name
                stage_warnings.append(f"{name}: sync check {item['status']}: {item['reason']}")
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
    changes: dict[str, Any] = {
        "placements": placements, "audio_warp_maps": warps, "outputs": outputs,
    }
    # The repaired recorder replaces the original for every later stage.
    transcript_keys = {key: path for key, path in project.transcripts.items()
                       if not key.startswith("repaired:")}
    for recorder_id, repair in repairs.items():
        outputs[f"recorder:{recorder_id}"] = [repair.audio]
        transcript_keys[f"repaired:{recorder_id}"] = repair.transcript
        artifacts.extend([repair.audio, repair.transcript])
    if transcript_keys != project.transcripts:
        changes["transcripts"] = transcript_keys
    return StageOutput(tuple(artifacts), changes, tuple(stage_warnings))


def _run_camera_placement(context: StageContext) -> StageOutput:
    from studio.stages.sync_media import selected_media

    context.work_dir.mkdir(parents=True, exist_ok=True)
    with selected_media(context.project, context.work_dir,
                        context.settings.get("transcribe", {}).get("audio_streams")) as project:
        return _place_cameras(StageContext(project, context.settings, context.work_dir,
                                           context.progress))


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


def _transcript_or_probe(project: Project, asset: Asset) -> Transcript:
    path = project.transcripts.get(asset.id)
    if path is not None and path.is_file():
        return Transcript.load(path)
    return Transcript(project.source_dir / asset.path, "", _duration(asset), [])


def refine_settings(sync: dict[str, Any], camera: Asset) -> RefineSettings | None:
    """Acoustic plan refinement for ``camera``, or ``None`` when off or it has no sound."""
    if sync.get("acoustic_refine", "auto") == "off":
        return None
    info = camera.manual.get("media_info", {})
    if info and info.get("audio_stream_index") is None and not info.get("audio_channels"):
        return None
    return RefineSettings.from_sync(sync, piece_config(sync).seam_snap_max_s)


def self_check_transcribe(settings: dict[str, Any]) -> dict[str, Any]:
    """Transcription settings of the self-check: sync.self_check_transcribe_mode
    replaces transcribe.mode when set (WhisperSync checked in "fast")."""
    transcribe = dict(settings.get("transcribe", {}))
    mode = str(settings.get("sync", {}).get("self_check_transcribe_mode") or "")
    return {**transcribe, "mode": mode} if mode else transcribe


def piece_config(sync: dict[str, Any]) -> PieceConfig:
    """Pieces cut in pauses, seams snapped within ``sync.seam_snap_max_s``."""
    return PieceConfig(seam_snap_max_s=float(sync.get("seam_snap_max_s", 0.4)))


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
