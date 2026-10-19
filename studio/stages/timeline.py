"""Build a timeline-domain transcript without another Whisper pass."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from studio.core.channels import effective_audio
from studio.core.project import AssetRole, Project, describe_artifact, stable_fingerprint
from studio.core.timeline import SourcePlacement, TimeDomain, file_to_timeline
from studio.core.transcript import Segment, Transcript, Word
from studio.core.voice_fill import (
    PLAYBACK_PREFIX,
    AudioFormat,
    audio_format,
    fill_voice,
    level_gain_db,
    merge,
    runs_seconds,
    subtract,
    unheard_spans,
    zero_runs,
)
from studio.stages.base import (
    Decision,
    GpuUse,
    Requirement,
    StageContext,
    StageOutput,
    select_settings,
)


class TimelineStage:
    id = "timeline"
    title = "Timeline transcript"
    after: tuple[str, ...] = ("transcribe", "sync")
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("sync.mode", "sync.reference_camera",
                                     "sync.recorder_fill")
    optional_after = ("sync",)
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return []

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        source_id = _transcript_source_id(project, settings)
        if source_id is None:
            return Decision.skip("no transcript source is available")
        if source_id not in project.transcripts:
            return Decision.blocked("primary transcript is missing", "run transcription")
        return Decision.run({"source_asset_id": source_id})

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        source_id = _transcript_source_id(project, settings)
        transcript = _transcript_path(project, source_id or "")
        content = (
            transcript.read_text(encoding="utf-8")
            if transcript and transcript.is_file()
            else ""
        )
        placement = next(
            (item for item in project.placements if item.asset_id == source_id), None
        )
        session_inputs = {
            asset.id: path.read_text(encoding="utf-8")
            for asset in project.assets if asset.role is AssetRole.RECORDER
            for path in [_transcript_path(project, asset.id)]
            if path is not None and path.is_file()
        }
        # Cameras stand in where no recorder heard anything: their words and sound.
        camera_inputs = [
            describe_artifact(project.work_dir, path, project.work_dir)
            for asset in project.assets if asset.role is AssetRole.CAMERA
            for path in [project.transcripts.get(asset.id),
                         next(iter(project.outputs.get(f"sync:{asset.id}", [])), None)]
            if path is not None and path.is_file() and path.is_relative_to(project.work_dir)
        ]
        recorders = [
            describe_artifact(project.work_dir, path, project.work_dir)
            for asset in project.assets if asset.role is AssetRole.RECORDER
            for path in project.outputs.get(f"recorder:{asset.id}", [])
            if path.is_file() and path.is_relative_to(project.work_dir)
        ]
        return stable_fingerprint("timeline-v5", source_id, content, placement,
                                  session_inputs, project.placements, camera_inputs, recorders,
                                  select_settings(settings, self.settings_keys))

    def run(self, context: StageContext) -> StageOutput:
        source_id = _transcript_source_id(context.project, context.settings)
        if source_id is None:
            raise ValueError("no transcript source is available")
        source = Transcript.load(_transcript_path(context.project, source_id)
                                 or context.project.transcripts[source_id])
        placement = next(
            (item for item in context.project.placements if item.asset_id == source_id),
            _identity_placement(source_id, source.duration),
        )
        segments = [
            Segment(
                file_to_timeline(segment.start, placement),
                file_to_timeline(segment.end, placement),
                tuple(
                    Word(
                        word.text,
                        file_to_timeline(word.start, placement),
                        file_to_timeline(word.end, placement),
                        word.probability,
                    )
                    for word in segment.words
                ),
                segment.confidence,
                segment.avg_logprob,
                segment.text_override,
            )
            for segment in source.segments
        ]
        transcript = Transcript(
            source.source_audio,
            source.language,
            max((segment.end for segment in segments), default=0.0),
            segments,
            source.model,
            source.device,
            source.compute_type,
            source.mode,
            TimeDomain.TIMELINE,
            f"placement:{source_id}",
            metadata={key: value for key, value in source.metadata.items()
                      if key != "coverage"} | {"source_asset_id": source_id},
        )
        # Recorder transcripts made from tiles cover only part of their file:
        # consumers (roughcut) must not read the rest as untranscribed sound.
        coverage = [_mapped_coverage(source, placement)]
        used_placements = [placement]
        # Add disjoint recorder coverage without duplicating overlapping microphone speech.
        occupied = [(placement.offset_s,
                     placement.offset_s + placement.duration_s * placement.k)]
        source_ids = [source_id]
        for asset in context.project.assets:
            if context.settings.get("sync", {}).get("mode") == "camera":
                break
            if asset.role is not AssetRole.RECORDER or asset.id == source_id:
                continue
            path = _transcript_path(context.project, asset.id)
            mapped = next((item for item in context.project.placements
                           if item.asset_id == asset.id), None)
            if path is None or mapped is None:
                continue
            extra = Transcript.load(path)
            for segment in extra.segments:
                start = file_to_timeline(segment.start, mapped)
                end = file_to_timeline(segment.end, mapped)
                if any(start < right and end > left for left, right in occupied):
                    continue
                segments.append(Segment(start, end, tuple(
                    Word(word.text, file_to_timeline(word.start, mapped),
                         file_to_timeline(word.end, mapped), word.probability)
                    for word in segment.words
                ), segment.confidence, segment.avg_logprob, segment.text_override))
            occupied.append((mapped.offset_s, mapped.offset_s + mapped.duration_s * mapped.k))
            source_ids.append(asset.id)
            coverage.append(_mapped_coverage(extra, mapped))
            used_placements.append(mapped)
        segments.sort(key=lambda item: item.start)
        transcript.duration = max((segment.end for segment in segments), default=0.0)
        if len(source_ids) > 1:
            transcript.metadata["source_asset_ids"] = source_ids
        if any(item is not None for item in coverage):
            spans = [span for item in coverage if item is not None for span in item]
            # A whole-file source among partial ones covers its whole placement.
            spans += [(item.offset_s, item.offset_s + item.duration_s * item.k)
                      for item, mapped in zip(used_placements, coverage, strict=True)
                      if mapped is None]
            transcript.metadata["coverage"] = [[round(a, 3), round(b, 3)]
                                               for a, b in sorted(spans)]
        sync = context.settings.get("sync", {})
        sync = sync if isinstance(sync, dict) else {}
        outputs = {key: paths for key, paths in context.project.outputs.items()
                   if not key.startswith(PLAYBACK_PREFIX)}
        artifacts: list[Path] = []
        warnings: list[str] = []
        if (sync.get("recorder_fill", "camera") == "camera" and sync.get("mode") != "camera"
                and source_id in _recorder_ids(context.project)):
            unheard = unheard_spans(context.project)
            added, filled = _camera_words(context.project, transcript, unheard)
            if added:
                segments.extend(added)
                segments.sort(key=lambda item: item.start)
                transcript.duration = max(transcript.duration, max(seg.end for seg in added))
                if "coverage" in transcript.metadata:
                    transcript.metadata["coverage"] = [
                        [round(a, 3), round(b, 3)] for a, b in merge(
                            [(float(a), float(b)) for a, b in transcript.metadata["coverage"]]
                            + [(item["start"], item["end"]) for item in filled])]
            voices, gains = _playback_voices(context, warnings)
            for item in filled:
                item["gain_db"] = gains.get(str(item["asset_id"]))
            if filled:
                transcript.metadata["camera_fill"] = filled
            for asset_id, path in voices.items():
                outputs[f"{PLAYBACK_PREFIX}{asset_id}"] = [path]
                artifacts.append(path)
            if unheard:
                total = sum(end - start for start, end in unheard)
                warnings.append(
                    f"no recorder heard {total:.1f} s under the cameras in {len(unheard)} "
                    f"place(s); the cameras' sound fills {len(voices)} voice(s)"
                    + (f" and their words {sum(len(seg.words) for seg in added)} word(s) "
                       "of the transcript" if added else ""))
        output = context.work_dir / "stages" / "timeline" / "transcript.json"
        transcript.save(output)
        transcripts = {**context.project.transcripts, "timeline": output}
        changes: dict[str, Any] = {"transcripts": transcripts, "outputs": outputs}
        cameras = [asset for asset in context.project.assets if asset.role is AssetRole.CAMERA]
        if len(cameras) == 1 and not context.project.placements and source_id == cameras[0].id:
            changes["placements"] = [placement]
        return StageOutput((output, *artifacts), changes, tuple(warnings))


def _recorder_ids(project: Project) -> set[str]:
    return {asset.id for asset in project.assets if asset.role is AssetRole.RECORDER}


def _camera_words(
    project: Project, transcript: Transcript, unheard: list[tuple[float, float]],
) -> tuple[list[Segment], list[dict[str, Any]]]:
    """Camera words for the stretches no recorder heard, best-transcribed camera first.

    Within each stretch the camera with the most words there is taken, then the
    next one for what it leaves uncovered, never over a word already present.
    """
    placements = {item.asset_id: item for item in project.placements}
    sources: list[tuple[str, SourcePlacement, Transcript]] = []
    for asset in project.assets:
        path = project.transcripts.get(asset.id)
        placement = placements.get(asset.id)
        if (asset.role is not AssetRole.CAMERA or placement is None or path is None
                or not path.is_file()):
            continue
        camera = Transcript.load(path)
        if camera.time_domain is TimeDomain.FILE:
            sources.append((asset.id, placement, camera))
    present = [(word.start, word.end) for word in transcript.words]
    added: list[Segment] = []
    filled: list[dict[str, Any]] = []
    for stretch in unheard:
        remaining = [stretch]
        while remaining:
            best: tuple[int, str, list[Segment], list[tuple[float, float]]] | None = None
            for asset_id, placement, camera in sources:
                span = (placement.offset_s, placement.offset_s + placement.duration_s * placement.k)
                parts = [(max(a, span[0]), min(b, span[1])) for a, b in remaining
                         if min(b, span[1]) > max(a, span[0])]
                if not parts:
                    continue
                segments = _segments_inside(camera, placement, parts, present)
                count = sum(len(segment.words) for segment in segments)
                if count and (best is None or count > best[0]):
                    best = (count, asset_id, segments, parts)
            if best is None:
                break
            _count, asset_id, segments, parts = best
            added.extend(segments)
            present.extend((word.start, word.end) for seg in segments for word in seg.words)
            filled.extend({"start": round(a, 3), "end": round(b, 3), "asset_id": asset_id}
                          for a, b in parts)
            remaining = subtract(remaining, parts)
    filled.sort(key=lambda item: item["start"])
    return added, filled


def _segments_inside(
    camera: Transcript, placement: SourcePlacement, parts: list[tuple[float, float]],
    present: list[tuple[float, float]],
) -> list[Segment]:
    result: list[Segment] = []
    for segment in camera.segments:
        words = []
        for word in segment.words:
            start = file_to_timeline(word.start, placement)
            end = file_to_timeline(word.end, placement)
            middle = (start + end) / 2
            if (any(a <= middle < b for a, b in parts)
                    and not any(start < right and end > left for left, right in present)):
                words.append(Word(word.text, start, end, word.probability))
        if words:
            whole = len(words) == len(segment.words)
            result.append(Segment(words[0].start, words[-1].end, tuple(words), segment.confidence,
                                  segment.avg_logprob,
                                  segment.text_override if whole else None))
    return result


def _playback_voices(
    context: StageContext, warnings: list[str],
) -> tuple[dict[str, Path], dict[str, float]]:
    """Camera voices with the camera's own sound where they are digital silence.

    A camera whose level can't be measured (no stretch where both are audible)
    takes the median gain of its camera group, then of all cameras.
    """
    project = context.project
    pending: list[tuple[Any, Path, list[tuple[int, int]], AudioFormat]] = []
    gains: dict[str, float] = {}
    for asset in project.assets:
        voice = next(iter(project.outputs.get(f"sync:{asset.id}", [])), None)
        if (asset.role is not AssetRole.CAMERA or voice is None or not voice.is_file()
                or not asset.manual.get("media_info", {}).get("audio_codec")):
            continue
        camera = project.source_dir / asset.path
        try:
            audio = audio_format(voice)
            runs = zero_runs(voice, audio=audio)
            if not runs:
                continue
            stream = asset.manual.get("media_info", {}).get("audio_stream_index")
            gain = level_gain_db(voice, camera, runs_seconds(runs, audio), stream=stream,
                                 pan=effective_audio(asset).pan)
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"{Path(asset.path).name}: sound not filled: {exc}")
            continue
        if gain is not None:
            gains[asset.id] = gain
        pending.append((asset, voice, runs, audio))
    voices: dict[str, Path] = {}
    for asset, voice, runs, audio in pending:
        group = [gain for other in project.assets if other.group_id == asset.group_id
                 for gain in [gains.get(other.id)] if gain is not None]
        gain = gains.get(asset.id)
        if gain is None:
            fallback = group or list(gains.values())
            gain = float(np.median(fallback)) if fallback else 0.0
            gains[asset.id] = gain
        output = context.work_dir / "stages" / "timeline" / "voice" / f"{asset.id}.wav"
        info = asset.manual.get("media_info", {})
        try:
            voices[asset.id] = fill_voice(
                voice, project.source_dir / asset.path, output, runs, gain_db=gain,
                stream=info.get("audio_stream_index"), pan=effective_audio(asset).pan,
                audio=audio)
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"{Path(asset.path).name}: sound not filled: {exc}")
    return voices, {key: round(value, 2) for key, value in gains.items()}


def _transcript_path(project: Project, asset_id: str) -> Path | None:
    """A recorder's transcript, on its repaired clock when sync repaired it."""
    return project.transcripts.get(f"repaired:{asset_id}") or project.transcripts.get(asset_id)


def _mapped_coverage(
    transcript: Transcript, placement: SourcePlacement
) -> list[tuple[float, float]] | None:
    """A file transcript's ``metadata.coverage`` on the timeline (``None``: whole file)."""
    spans = transcript.metadata.get("coverage")
    if not spans:
        return None
    return [(file_to_timeline(float(a), placement), file_to_timeline(float(b), placement))
            for a, b in spans]


def _primary_source_id(project: Project) -> str | None:
    for role in (AssetRole.RECORDER, AssetRole.CAMERA):
        source = next((asset.id for asset in project.assets if asset.role is role), None)
        if source is not None:
            return source
    return None


def _transcript_source_id(project: Project, settings: dict[str, Any]) -> str | None:
    if settings.get("sync", {}).get("mode") == "camera":
        from studio.stages.sync import _ordered_cameras

        cameras = _ordered_cameras(project, settings)
        return cameras[0].id if cameras else None
    return _primary_source_id(project)


def _identity_placement(asset_id: str, duration: float) -> SourcePlacement:
    return SourcePlacement(asset_id, 0.0, 0.0, max(duration, 1e-6), provenance="metadata")
