"""Render verified moments in their declared transcript time domain."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from studio.core.framing import load_reel_overrides, overrides_path, reel_settings
from studio.core.launch import frozen_root
from studio.core.media import probe
from studio.core.proc import run_logged
from studio.core.project import Project, stable_fingerprint
from studio.core.subtitles import load_edits, to_ass
from studio.core.timeline import EditMap, KeepRange, TimeDomain
from studio.core.transcript import Segment, Transcript, Word, to_srt
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.program import render_project_program
from studio.stages.reel_finish import normalize_loudness, render_cover
from studio.stages.reel_layout import (
    compose_title,
    reel_title,
    render_audiogram,
    title_band_height,
)
from studio.stages.reels import _settings, _transcript_path


class ReelRenderStage:
    id = "reel_render"
    title = "Render reels"
    after: tuple[str, ...] = ("reels", "program")
    optional_after = ("program",)
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        if not _settings(settings).get("tracking", False):
            return [Requirement("binary", "ffmpeg", "render reels")]
        # Tracking falls back to a static crop, so vision pieces are optional.
        return [
            Requirement("binary", "ffmpeg", "render reels"),
            Requirement("module", "vision", "face tracking", optional=True),
            Requirement("model", "yunet", "face detection for tracking", optional=True),
            Requirement("model", "light-asd", "active speaker for tracking", optional=True),
        ]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        if not _settings(settings).get("render", False):
            return Decision.skip("reels rendering is disabled")
        if not project.outputs.get("reels"):
            return Decision.blocked("no verified moments are available", "run reels")
        path = _transcript_path(project)
        if path is None:
            return Decision.blocked("no transcript is available", "run timeline")
        if Transcript.load(path).time_domain is TimeDomain.EDITED and not project.outputs.get(
            "program"
        ):
            return Decision.blocked("edited moments need the review program", "enable program")
        return Decision.run({})

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        inputs = [*project.outputs.get("reels", []), *project.outputs.get("program", []),
                  *project.outputs.get("speakers", [])]
        inputs.extend(project.source_dir / asset.path for asset in project.assets)
        inputs.extend(
            path
            for key, paths in project.outputs.items()
            if key.startswith("sync:")
            for path in paths
        )
        transcript = _transcript_path(project)
        if transcript:
            inputs.append(transcript)
        overrides = overrides_path(project.work_dir)
        if overrides.is_file():
            inputs.append(overrides)
        saved = project.work_dir / "subtitles"
        if saved.is_dir():
            inputs.extend(sorted(saved.glob("*.json")))
        model_dir = Path(
            os.environ.get(
                "UVF_MODELS_DIR", str(Path.home() / ".cache" / "ultimate-video-forge" / "models")
            )
        )
        if _settings(settings).get("tracking", False):
            inputs.extend(
                model_dir / name
                for name in ("face_detection_yunet_2023mar.onnx", "light_asd_talkset.model")
            )
        return stable_fingerprint(
            "reel-render-v6",
            _settings(settings),
            settings.get("program", {}), project.speaker_cameras,
            project.to_dict()["assets"],
            [str(placement) for placement in project.placements],
            {
                key: [str(path) for path in paths]
                for key, paths in project.outputs.items()
                if key.startswith("sync:")
            },
            [(str(path), _digest(path)) for path in inputs if path.is_file()],
        )

    def run(self, context: StageContext) -> StageOutput:
        settings = _settings(context.settings)
        transcript_path = _transcript_path(context.project)
        if transcript_path is None:
            raise ValueError("no transcript is available")
        transcript = Transcript.load(transcript_path)
        selection_report = next(
            (
                path
                for path in context.project.outputs.get("reels", [])
                if path.name == "report.json"
            ),
            None,
        )
        if selection_report:
            selection = json.loads(selection_report.read_text(encoding="utf-8"))
            if selection.get("time_domain") != transcript.time_domain.value or selection.get(
                "transcript_fingerprint"
            ) != stable_fingerprint(transcript.to_dict()):
                raise ValueError("moments belong to a different transcript; rerun reels selection")
        moments_path = next(
            path for path in context.project.outputs.get("reels", []) if path.name == "moments.json"
        )
        moments = json.loads(moments_path.read_text(encoding="utf-8"))
        directory = context.work_dir / "stages" / "reel_render"
        directory.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = []
        entries = []
        overrides = load_reel_overrides(context.project.work_dir)
        fps = str(settings.get("render_fps", "25"))
        for number, moment in enumerate(moments, 1):
            start, end = float(moment["start"]), float(moment["end"])
            if not 0 <= start < end <= transcript.duration:
                raise ValueError("moment is outside its transcript")
            stem = f"reel-{number:03d}"
            key = str(moment.get("candidate_id") or stem)
            override = overrides.get(key, overrides.get(stem))
            reel = reel_settings(settings, override)
            layout = str(reel.get("layout", "video"))
            title_override = (override or {}).get("title")
            title = reel_title(moment, reel, str(title_override) if title_override else None)
            video = directory / f"{stem}.mp4"
            frame, source_size = None, None
            framing_report: dict[str, object] = {"tracking": "disabled"}
            if layout == "audiogram":
                with tempfile.TemporaryDirectory(prefix="uvf-reel-audio-") as folder:
                    audio = Path(folder) / "audio.wav"
                    reel_audio(context.project, transcript, start, end, audio)
                    render_audiogram(audio, video, title, end - start, reel)
                framing = "audiogram"
            else:
                render_reel_source(context.project, context.settings, transcript,
                                   start, end, stem, video, fps)
                # The unframed still lets the editor preview any crop instantly.
                frame = directory / f"{stem}-frame.jpg"
                render_cover(video, frame, (end - start) / 3)
                artifacts.append(frame)
                info = probe(video)
                source_size = [info.width, info.height]
                framing = str(reel.get("framing", "source"))
                framed = directory / f"{stem}-framed.mp4"
                if layout == "title":
                    area = {**reel, "height": int(reel.get("height", 1920))
                            - title_band_height(reel)}
                    if framing == "source":
                        area["framing"] = "fit"
                    framing_report = reframe_video(video, framed, area)
                    compose_title(framed, video, title, end - start, reel)
                    framed.unlink()
                elif framing != "source":
                    framing_report = reframe_video(video, framed, reel)
                    framed.replace(video)
            lufs = float(settings.get("loudness_lufs", -14.0))
            loudness = normalize_loudness(video, lufs) if lufs else {"status": "source"}
            cover = None
            if settings.get("covers", True):
                cover = directory / f"{stem}-cover.jpg"
                render_cover(video, cover, (end - start) / 3)
                artifacts.append(cover)
            published_video = context.project.work_dir / video.relative_to(context.work_dir)
            local = clip_transcript(transcript, start, end, published_video)
            cues, style = load_edits(context.project.work_dir, stem, local)
            subtitle_transcript = replace(
                local,
                segments=[Segment(cue.start, cue.end, text_override=cue.text) for cue in cues],
            )
            json_path, srt_path = directory / f"{stem}.json", directory / f"{stem}.srt"
            local.save(json_path)
            srt_path.write_text(to_srt(subtitle_transcript), encoding="utf-8")
            artifacts.extend((video, json_path, srt_path))
            ass_path = directory / f"{stem}.ass"
            ass_path.write_text(
                to_ass(
                    cues,
                    style,
                    portrait=(framing != "source" or layout != "video")
                    and int(settings.get("height", 1920)) > int(settings.get("width", 1080)),
                ),
                encoding="utf-8",
            )
            artifacts.append(ass_path)
            if settings.get("burn_subtitles", False):
                captioned = directory / f"{stem}-captioned.mp4"
                burn_subtitles(video, ass_path, captioned)
                artifacts.append(captioned)
            entries.append(
                {
                    "candidate_id": moment.get("candidate_id"),
                    "video": video.name,
                    "transcript": json_path.name,
                    "subtitles": srt_path.name,
                    "styled_subtitles": ass_path.name,
                    "captioned": f"{stem}-captioned.mp4"
                    if settings.get("burn_subtitles")
                    else None,
                    "source_start": start,
                    "source_end": end,
                    "source_time_domain": transcript.time_domain.value,
                    "reel_key": key,
                    "layout": layout,
                    "title": title if layout != "video" else None,
                    "framing": framing,
                    "framing_report": framing_report,
                    "crop_x": reel.get("crop_x", 0.5) if framing == "crop" else None,
                    "frame": frame.name if frame else None,
                    "source_size": source_size,
                    "override": bool(override),
                    "loudness": loudness,
                    "cover": cover.name if cover else None,
                }
            )
        report = directory / "render.json"
        report.write_text(json.dumps({"schema_version": 1, "reels": entries}, indent=2) + "\n")
        artifacts.append(report)
        return StageOutput(
            tuple(artifacts),
            {
                "outputs": {**context.project.outputs, "reel_render": artifacts},
            },
        )


def render_reel_source(
    project: Project, all_settings: dict[str, Any], transcript: Transcript,
    start: float, end: float, stem: str, video: Path, fps: str,
) -> None:
    """The reel's unframed picture and sound from its transcript's time domain."""
    if transcript.time_domain is TimeDomain.TIMELINE:
        render_project_program(
            project,
            reel_camera_plan(project, all_settings,
                             EditMap(stem, (KeepRange(start, end),))),
            video,
            encoder="cpu",
            fps=fps,
        )
    elif transcript.time_domain is TimeDomain.EDITED:
        sources = project.outputs.get("program", [])
        if not sources:
            raise ValueError("edited moments require a rendered program")
        result = run_logged(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(sources[0]),
                "-t",
                str(end - start),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                f"fps={fps}",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(video),
            ],
            timeout=max(120, (end - start) * 20),
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
    else:
        raise ValueError("reels require timeline or edited transcript times")


def reel_audio(
    project: Project, transcript: Transcript, start: float, end: float, output: Path
) -> None:
    """The reel's sound alone, for layouts that need no camera."""
    if transcript.time_domain is TimeDomain.TIMELINE:
        from studio.core.preview import render_preview

        render_preview(project, [(start, end)], output)
        return
    if transcript.time_domain is not TimeDomain.EDITED:
        raise ValueError("reels require timeline or edited transcript times")
    sources = project.outputs.get("program", [])
    if not sources:
        raise ValueError("edited moments require a rendered program")
    result = run_logged(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(sources[0]),
         "-t", str(end - start), "-map", "0:a:0", "-ar", "48000", str(output)],
        timeout=max(120, (end - start) * 10),
    )
    if result.returncode:
        raise RuntimeError(result.stderr)


def clip_transcript(source: Transcript, start: float, end: float, video: Path) -> Transcript:
    segments = []
    for segment in source.segments:
        if segment.end <= start or segment.start >= end:
            continue
        words = tuple(
            Word(
                word.text,
                max(start, word.start) - start,
                min(end, word.end) - start,
                word.probability,
            )
            for word in segment.words
            if word.start < end and word.end > start
        )
        segments.append(
            Segment(
                max(start, segment.start) - start,
                min(end, segment.end) - start,
                words,
                text_override=None if words else segment.text,
            )
        )
    return replace(
        source,
        source_audio=video,
        duration=end - start,
        segments=segments,
        time_domain=TimeDomain.FILE,
        map_id=None,
        metadata={
            **source.metadata,
            "reel_source_start": start,
            "reel_source_time_domain": source.time_domain.value,
        },
    )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def burn_subtitles(video: Path, subtitles: Path, output: Path) -> None:
    # Keep arbitrary project filenames out of ffmpeg's filtergraph parser.
    with tempfile.TemporaryDirectory(prefix="uvf-subtitles-") as temporary:
        safe_path = Path(temporary) / f"captions{subtitles.suffix}"
        shutil.copyfile(subtitles, safe_path)
        escaped = str(safe_path).replace("\\", "/").replace(":", "\\:")
        result = run_logged(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                f"subtitles=filename='{escaped}'",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            timeout=3600,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)


def reframe_video(video: Path, output: Path, settings: dict[str, Any]) -> dict[str, object]:
    width, height = int(settings.get("width", 1080)), int(settings.get("height", 1920))
    if settings.get("framing") == "crop":
        x = float(settings.get("crop_x", 0.5))
        filters = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(iw-ow)*{x}:(ih-oh)/2,setsar=1"
        )
    else:
        filters = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    tracking = "disabled"
    if settings.get("tracking", False):
        tracked = tracking_filter(video, width, height, settings)
        tracking = "applied" if tracked else "unavailable-static-fallback"
        filters = tracked or filters
    result = run_logged(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            filters,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout=3600,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return {"tracking": tracking, "width": width, "height": height}


def reel_camera_plan(project: Project, settings: dict[str, Any], edit: EditMap) -> EditMap:
    """Switch reel angles by speaker with the program's camera policy, when configured."""
    from studio.core.recorder_tracks import effective_speaker_cameras
    from studio.stages.camera_selection import CameraPolicy, speaker_camera_edit

    conf = settings.get("program", {})
    conf = conf if isinstance(conf, dict) else {}
    if not _settings(settings).get("speaker_angles", True):
        return edit
    mapping = effective_speaker_cameras(project, conf.get("speaker_cameras", {}))
    paths = project.outputs.get("speakers", [])
    if not mapping or not paths or not paths[0].is_file():
        return edit
    turns = json.loads(paths[0].read_text(encoding="utf-8"))
    return speaker_camera_edit(project, edit, turns, mapping,
                               policy=CameraPolicy.from_settings(conf))


def run_vision_worker(request: dict[str, Any], timeout: float = 3600) -> dict[str, Any]:
    """Run one request through the managed vision interpreter."""
    from studio.modules.manager import ModuleManager

    manager = ModuleManager()
    if not manager.installed("vision"):
        raise ValueError("managed vision module is missing; install it through modules install")
    with tempfile.TemporaryDirectory(prefix="uvf-vision-") as folder:
        request_path, output = Path(folder) / "request.json", Path(folder) / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(
                frozen_root() / "managed"
                if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parents[2]
            )
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )
        result = run_logged(
            [str(manager.interpreter("vision")), "-m", "studio.vision.worker",
             str(request_path), str(output)],
            env=environment,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-4096:])
        value: dict[str, Any] = json.loads(output.read_text(encoding="utf-8"))
        return value


def tracking_filter(
    video: Path, width: int, height: int, settings: dict[str, Any]
) -> str | None:
    import logging

    try:
        from studio.core.media import probe

        value = run_vision_worker({
            "video": str(video.resolve()),
            "duration": probe(video).duration_s,
            "width": width,
            "height": height,
            "device": str(settings.get("tracking_device", "cuda")),
            "active_speaker": bool(settings.get("active_speaker", True)),
        })
        if value["filter"]:
            filter_expression: str = value["filter"]
            return filter_expression + ",setsar=1"
    except Exception as exc:
        logging.getLogger(__name__).warning("tracking unavailable: %s", exc)
    logging.getLogger(__name__).warning("tracking unavailable; using configured static crop")
    return None
