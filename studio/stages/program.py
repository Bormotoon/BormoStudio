"""Render a review master and derive its edited-time transcript."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from studio.core.media import build_atempo_chain
from studio.core.project import AssetRole, Project, describe_artifact, stable_fingerprint
from studio.core.timeline import (
    EditMap,
    KeepRange,
    TimeDomain,
    timeline_to_edited,
    timeline_to_file,
)
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.program_encoder import select_encoder


class ProgramStage:
    id = "program"
    title = "Review master"
    after: tuple[str, ...] = ("timeline", "roughcut", "speakers")
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("program", "roughcut")
    optional_after = ("roughcut", "speakers")
    gpu = GpuUse.NVENC

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return [Requirement("binary", "ffmpeg", "render the review master")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        program = settings.get("program", {})
        enabled = program.get("enabled", False) if isinstance(program, dict) else False
        return Decision.run() if enabled else Decision.skip("program render is disabled")

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        edit_path = _edit_path(project)
        edit = edit_path.read_text(encoding="utf-8") if edit_path.is_file() else ""
        return stable_fingerprint(
            "program-v8",
            edit,
            project.placements,
            project.outputs.get("sync"),
            [
                path.read_text(encoding="utf-8")
                for path in project.outputs.get("speakers", [])
                if path.is_file()
            ],
            [(asset.id, asset.group_id) for asset in project.assets],
            [
                describe_artifact(project.work_dir, path, project.work_dir)
                for key, paths in project.outputs.items()
                if key.startswith("sync:")
                for path in paths
                if path.is_file()
            ],
            [
                describe_artifact(project.source_dir, project.source_dir / asset.path,
                                  project.work_dir)
                for asset in project.assets
                if asset.role is AssetRole.CAMERA
            ],
            {
                key: value
                for key, value in settings.get("program", {}).items()
                if key != "burn_subtitles"
            },
            settings.get("roughcut", {}),
            project.speaker_cameras,
        )

    def run(self, context: StageContext) -> StageOutput:
        timeline = Transcript.load(context.project.transcripts["timeline"])
        output = context.work_dir / "program" / "program.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        conf = context.settings.get("program", {})
        conf = conf if isinstance(conf, dict) else {}
        edit = program_edit(context.project, context.settings, timeline.duration)
        render_report = render_project_program(
            context.project,
            edit,
            output,
            encoder=str(conf.get("encoder", "auto")),
            fps=str(conf.get("fps", "auto")),
            crossfade_ms=float(conf.get("crossfade_ms", 10.0)),
        )
        transcript_ranges: list[KeepRange] = []
        for item in edit.keep:
            if transcript_ranges and abs(transcript_ranges[-1].end_s - item.start_s) < 1e-9:
                transcript_ranges[-1] = KeepRange(transcript_ranges[-1].start_s, item.end_s)
            else:
                transcript_ranges.append(KeepRange(item.start_s, item.end_s))
        edited = map_transcript_to_edited(timeline, EditMap(edit.id, tuple(transcript_ranges)))
        edited = align_transcript_to_frames(edited, render_report)
        render_report["timeline_to_rendered"] = rendered_timeline_map(edit, render_report)
        render_report["edited_timing_fingerprint"] = stable_fingerprint(
            edited.duration, [(word.start, word.end) for word in edited.words]
        )
        render_report["camera_plan"] = [
            {"start_s": item.start_s, "end_s": item.end_s, "camera_id": item.camera_id}
            for item in edit.keep
        ]
        edited_path = context.work_dir / "program" / "program.transcript.json"
        edited.save(edited_path)
        report_path = output.with_name("render.json")
        report_path.write_text(json.dumps(render_report, indent=2) + "\n", encoding="utf-8")
        transcripts = {**context.project.transcripts, "edited": edited_path}
        outputs = {**context.project.outputs, "program": [output], "program_report": [report_path]}
        return StageOutput(
            (output, edited_path, report_path), {"transcripts": transcripts, "outputs": outputs}
        )


def program_edit(project: Project, settings: dict[str, Any], duration: float) -> EditMap:
    """The program's edit: kept rough-cut ranges, cut between speaker cameras."""
    edit_path = _edit_path(project)
    roughcut = settings.get("roughcut", {})
    use_edit = roughcut.get("enabled", True) if isinstance(roughcut, dict) else True
    edit = (
        load_edit_map(edit_path)
        if edit_path.is_file() and use_edit
        else EditMap("uncut", (KeepRange(0.0, duration),))
    )
    conf = settings.get("program", {})
    conf = conf if isinstance(conf, dict) else {}
    from studio.core.recorder_tracks import effective_speaker_cameras

    mapping = effective_speaker_cameras(project, conf.get("speaker_cameras", {}))
    if not mapping:
        return edit
    from studio.stages.camera_selection import CameraPolicy, speaker_camera_edit

    paths = project.outputs.get("speakers", [])
    if not paths or not paths[0].is_file():
        raise ValueError("speaker camera selection requires speakers output; run speakers")
    return speaker_camera_edit(
        project, edit, json.loads(paths[0].read_text(encoding="utf-8")), mapping,
        policy=CameraPolicy.from_settings(conf),
    )


def load_edit_map(path: Path) -> EditMap:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") == "markers":
        ranges = data.get("keep", [])
        if ranges:
            return EditMap(
                "uncut",
                (
                    KeepRange(
                        min(float(item["start"]) for item in ranges),
                        max(float(item["end"]) for item in ranges),
                    ),
                ),
            )
    return EditMap(
        "roughcut",
        tuple(
            KeepRange(float(item["start"]), float(item["end"]), item.get("camera_id"))
            for item in data.get("keep", [])
        ),
    )


@dataclass(frozen=True)
class ProgramPiece:
    video: Path
    audio: Path | None
    video_in_s: float
    audio_in_s: float
    duration_s: float
    rate: float
    timeline_start_s: float = 0.0


def program_pieces(project: Project, edit: EditMap) -> list[ProgramPiece]:
    cameras = {asset.id: asset for asset in project.assets if asset.role is AssetRole.CAMERA}
    placements = [item for item in project.placements if item.asset_id in cameras]
    result = []
    for keep in edit.keep:
        boundaries = sorted(
            {keep.start_s, keep.end_s}
            | {
                value
                for item in placements
                for value in (item.offset_s, item.offset_s + item.duration_s * item.k)
                if keep.start_s < value < keep.end_s
            }
        )
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            available = [
                item
                for item in placements
                if item.offset_s <= start + 1e-6
                and item.offset_s + item.duration_s * item.k >= end - 1e-6
            ]
            if keep.camera_id:
                group_ids = {
                    asset.id
                    for asset in cameras.values()
                    if asset.group_id == keep.camera_id or asset.id == keep.camera_id
                }
                available = [item for item in available if item.asset_id in group_ids]
            if not available:
                raise ValueError(f"no camera covers retained range {start:.3f}-{end:.3f}")
            placement = available[0]
            asset = cameras[placement.asset_id]
            source = project.source_dir / asset.path
            voice = next(iter(project.outputs.get(f"sync:{asset.id}", [])), None)
            has_audio = bool(asset.manual.get("media_info", {}).get("audio_codec"))
            audio = voice or (source if has_audio else None)
            source_in = timeline_to_file(start, placement)
            result.append(
                ProgramPiece(source, audio, source_in, source_in, end - start, placement.k, start)
            )
    if not result:
        raise ValueError("cannot render empty retained ranges")
    return result


def render_project_program(
    project: Project,
    edit: EditMap,
    output: Path,
    *,
    encoder: str = "auto",
    fps: str = "auto",
    crossfade_ms: float = 0.0,
    max_height: int | None = None,
) -> dict[str, object]:
    """Render ``edit``; ``max_height`` makes a fast low-resolution proxy."""
    pieces = program_pieces(project, edit)
    first = next(
        asset for asset in project.assets if project.source_dir / asset.path == pieces[0].video
    )
    info = first.manual.get("media_info", {})
    width, height = int(info.get("width") or 1920), int(info.get("height") or 1080)
    if max_height is not None and height > max_height:
        width, height = max(2, round(width * max_height / height / 2) * 2), max_height
    requested_fps = fps
    rate = Fraction(str(info.get("fps") or 25) if fps == "auto" else fps)
    if not 1 <= rate <= 240:
        raise ValueError("program fps must be between 1 and 240")
    fps = str(rate)
    # Quantize cumulative boundaries, not each duration independently. Many
    # sub-frame cuts must not accumulate a frame of error per edit.
    frame_counts = []
    cumulative = Fraction(0)
    previous_frame = 0
    for piece in pieces:
        cumulative += Fraction(str(piece.duration_s))
        end_frame = round(cumulative * rate)
        frame_counts.append(end_frame - previous_frame)
        previous_frame = end_frame
    if previous_frame == 0:
        raise ValueError("retained duration is shorter than one output frame")
    choice = select_encoder(encoder)
    codec = choice.codec
    fallback_reason = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as temporary:
        directory = Path(temporary)
        for index, piece in enumerate(pieces):
            frames = frame_counts[index]
            if frames == 0:
                continue
            rendered_duration = float(Fraction(frames) / rate)
            duration = piece.duration_s / piece.rate
            command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(piece.video)]
            if piece.audio is not None:
                command.extend(["-i", str(piece.audio)])
                audio = (
                    f"[1:a]atrim=start={piece.audio_in_s}:duration={duration},"
                    "asetpts=PTS-STARTPTS," + ",".join(build_atempo_chain(1 / piece.rate)) + ","
                    "aresample=48000,aformat=channel_layouts=stereo,"
                    f"apad,atrim=duration={rendered_duration}[a]"
                )
            else:
                audio = f"anullsrc=r=48000:cl=stereo,atrim=duration={rendered_duration}[a]"
            video = (
                f"[0:v]trim=start={piece.video_in_s}:duration={duration},"
                f"setpts={piece.rate}*(PTS-STARTPTS),"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                f"fps={fps},tpad=stop_mode=clone:stop_duration={rendered_duration},"
                f"trim=end_frame={frames},setpts=N/({fps}*TB)[v]"
            )
            command.extend(
                [
                    "-filter_complex",
                    video + ";" + audio,
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    "-c:v",
                    codec,
                    *choice.options,
                    *(["-preset", "ultrafast", "-crf", "30"]
                      if max_height is not None and codec == "libx264" else []),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "pcm_s16le",
                    "-t",
                    str(rendered_duration),
                    str(directory / f"part-{index}.mkv"),
                ]
            )
            try:
                _run_ffmpeg(command)
            except RuntimeError as exc:
                if encoder != "auto" or codec == "libx264":
                    raise
                # Regenerate every part to avoid joining different H.264
                # parameter sets when hardware fails halfway through a job.
                retry = render_project_program(
                    project, edit, output, encoder="cpu", fps=requested_fps,
                    crossfade_ms=crossfade_ms, max_height=max_height,
                )
                retry.update(
                    {
                        "requested_encoder": encoder,
                        "initial_codec": choice.codec,
                        "selection_reason": choice.reason,
                        "runtime_fallback_reason": str(exc)[-2000:],
                    }
                )
                return retry
        listing = directory / "parts.txt"
        listing.write_text(
            "".join(
                f"file 'part-{index}.mkv'\nduration {float(Fraction(frames) / rate):.12f}\n"
                for index, frames in enumerate(frame_counts)
                if frames
            ),
            encoding="utf-8",
        )
        audio_report = _mix_join_audio(pieces, frame_counts, rate, directory, crossfade_ms)
        mixed = ["-i", str(directory / "audio.wav"), "-map", "0:v", "-map", "1:a"]
        _run_ffmpeg(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                *(mixed if audio_report["audio_crossfade_ms"] else []),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output),
            ]
        )
    return {
        **audio_report,
        "schema_version": 1,
        "requested_encoder": encoder,
        "initial_codec": choice.codec,
        "final_codec": codec,
        "selection_reason": choice.reason,
        "runtime_fallback_reason": fallback_reason,
        "width": width,
        "height": height,
        "fps": fps,
        "piece_count": len(pieces),
        "requested_fps": requested_fps,
        "frame_count": previous_frame,
        "frame_counts": frame_counts,
        "timing_policy": "CFR cumulative frame boundaries",
        "piece_durations_s": [piece.duration_s for piece in pieces],
        "requested_duration_s": sum(piece.duration_s for piece in pieces),
        "duration_s": float(Fraction(previous_frame) / rate),
    }


def _mix_join_audio(
    pieces: list[ProgramPiece], frame_counts: list[int], rate: Fraction,
    directory: Path, crossfade_ms: float,
) -> dict[str, object]:
    """Replace hard audio joins with handled crossfades when requested."""
    from studio.stages.program_audio import SAMPLE_RATE, PartAudio, mix_parts

    parts: list[PartAudio] = []
    cumulative = 0
    previous: ProgramPiece | None = None
    for piece, frames in zip(pieces, frame_counts, strict=True):
        if not frames:
            continue
        start = round(Fraction(cumulative) / rate * SAMPLE_RATE)
        cumulative += frames
        samples = round(Fraction(cumulative) / rate * SAMPLE_RATE) - start
        continuous = previous is not None and abs(
            previous.timeline_start_s + previous.duration_s - piece.timeline_start_s
        ) < 1e-6
        parts.append(PartAudio(piece.audio, piece.audio_in_s, piece.rate, samples, continuous))
        previous = piece
    if crossfade_ms <= 0 or len(parts) < 2:
        return {"audio_crossfade_ms": 0.0, "audio_join_policy": "hard cut"}
    return mix_parts(parts, directory / "audio.wav", crossfade_ms)


def rendered_timeline_map(edit: EditMap, report: dict) -> list[dict]:
    """Intersect retained timeline spans with actual CFR trim/pad intervals."""
    rate = float(Fraction(str(report["fps"])))
    retained = []
    cursor = 0.0
    for keep in edit.keep:
        duration = keep.end_s - keep.start_s
        retained.append((cursor, cursor + duration, keep.start_s))
        cursor += duration
    result = []
    original = rendered = 0.0
    for duration, frames in zip(report["piece_durations_s"], report["frame_counts"], strict=True):
        output_duration = frames / rate
        for start, end, timeline in retained:
            left, right = (
                max(original, start),
                min(original + duration, end, original + output_duration),
            )
            if right > left:
                result.append(
                    {
                        "timeline_start": timeline + left - start,
                        "timeline_end": timeline + right - start,
                        "edited_start": rendered + left - original,
                        "edited_end": rendered + right - original,
                    }
                )
        original += duration
        rendered += output_duration
    return result


def align_transcript_to_frames(transcript: Transcript, report: dict[str, Any]) -> Transcript:
    """Apply frame-boundary trim/pad decisions to edited-time word positions."""
    rate = float(Fraction(str(report["fps"])))
    intervals = []
    original_cursor = rendered_cursor = 0.0
    for duration, frames in zip(report["piece_durations_s"], report["frame_counts"], strict=True):
        rendered_duration = frames / rate
        intervals.append(
            (original_cursor, original_cursor + duration, rendered_cursor, rendered_duration)
        )
        original_cursor += duration
        rendered_cursor += rendered_duration
    segments = []
    for segment in transcript.segments:
        words = []
        for word in segment.words:
            mapped = []
            for start, end, target, duration in intervals:
                left, right = max(start, word.start), min(end, word.end, start + duration)
                if right > left:
                    mapped.append((target + left - start, target + right - start))
            if mapped:
                words.append(Word(word.text, mapped[0][0], mapped[-1][1], word.probability))
        if words:
            segments.append(
                Segment(
                    words[0].start,
                    words[-1].end,
                    tuple(words),
                    segment.confidence,
                    text_override=segment.text_override,
                )
            )
    from dataclasses import replace

    return replace(transcript, duration=rendered_cursor, segments=segments)


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"program render failed: {result.stderr.strip()}")


def map_transcript_to_edited(transcript: Transcript, edit: EditMap) -> Transcript:
    segments: list[Segment] = []
    for segment in transcript.segments:
        words: list[Word] = []
        for word in segment.words:
            if not any(item.start_s <= word.start and word.end <= item.end_s for item in edit.keep):
                continue
            start = timeline_to_edited(word.start, edit)
            end = timeline_to_edited(word.end, edit)
            if start is not None and end is not None:
                words.append(Word(word.text, start, end, word.probability))
        if words:
            segments.append(
                Segment(
                    words[0].start,
                    words[-1].end,
                    tuple(words),
                    segment.confidence,
                    text_override=segment.text_override,
                )
            )
    duration = sum(item.end_s - item.start_s for item in edit.keep)
    return Transcript(
        transcript.source_audio,
        transcript.language,
        duration,
        segments,
        transcript.model,
        transcript.device,
        transcript.compute_type,
        transcript.mode,
        TimeDomain.EDITED,
        edit.id,
    )


def render_program(source: Path, edit: EditMap, output: Path) -> None:
    if not edit.keep:
        raise ValueError("cannot render a program with no retained ranges")
    filters: list[str] = []
    inputs: list[str] = []
    for index, item in enumerate(edit.keep):
        filters.extend(
            [
                f"[0:v]trim=start={item.start_s}:end={item.end_s},setpts=PTS-STARTPTS[v{index}]",
                f"[0:a]atrim=start={item.start_s}:end={item.end_s},asetpts=PTS-STARTPTS[a{index}]",
            ]
        )
        inputs.append(f"[v{index}][a{index}]")
    filters.append(f"{''.join(inputs)}concat=n={len(edit.keep)}:v=1:a=1[v][a]")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"program render failed: {result.stderr.strip()}")


def _edit_path(project: Project) -> Path:
    outputs = project.outputs.get("roughcut", [])
    return outputs[0] if outputs else project.work_dir / "stages" / "roughcut" / "edit.json"
