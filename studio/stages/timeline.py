"""Build a timeline-domain transcript without another Whisper pass."""

from __future__ import annotations

from typing import Any

from studio.core.project import AssetRole, Project, stable_fingerprint
from studio.core.timeline import SourcePlacement, TimeDomain, file_to_timeline
from studio.core.transcript import Segment, Transcript, Word
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
    settings_keys: tuple[str, ...] = ("sync.mode", "sync.reference_camera")
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
        transcript = project.transcripts.get(source_id or "")
        content = (
            transcript.read_text(encoding="utf-8")
            if transcript and transcript.is_file()
            else ""
        )
        placement = next(
            (item for item in project.placements if item.asset_id == source_id), None
        )
        session_inputs = {
            asset.id: project.transcripts[asset.id].read_text(encoding="utf-8")
            for asset in project.assets if asset.role is AssetRole.RECORDER
            and asset.id in project.transcripts and project.transcripts[asset.id].is_file()
        }
        return stable_fingerprint("timeline-v3", source_id, content, placement,
                                  session_inputs, project.placements,
                                  select_settings(settings, self.settings_keys))

    def run(self, context: StageContext) -> StageOutput:
        source_id = _transcript_source_id(context.project, context.settings)
        if source_id is None:
            raise ValueError("no transcript source is available")
        source = Transcript.load(context.project.transcripts[source_id])
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
            metadata={**source.metadata, "source_asset_id": source_id},
        )
        # Add disjoint recorder coverage without duplicating overlapping microphone speech.
        occupied = [(placement.offset_s,
                     placement.offset_s + placement.duration_s * placement.k)]
        source_ids = [source_id]
        for asset in context.project.assets:
            if context.settings.get("sync", {}).get("mode") == "camera":
                break
            if asset.role is not AssetRole.RECORDER or asset.id == source_id:
                continue
            path = context.project.transcripts.get(asset.id)
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
        segments.sort(key=lambda item: item.start)
        transcript.duration = max((segment.end for segment in segments), default=0.0)
        if len(source_ids) > 1:
            transcript.metadata["source_asset_ids"] = source_ids
        output = context.work_dir / "stages" / "timeline" / "transcript.json"
        transcript.save(output)
        transcripts = {**context.project.transcripts, "timeline": output}
        changes: dict[str, Any] = {"transcripts": transcripts}
        cameras = [asset for asset in context.project.assets if asset.role is AssetRole.CAMERA]
        if len(cameras) == 1 and not context.project.placements and source_id == cameras[0].id:
            changes["placements"] = [placement]
        return StageOutput((output,), changes)


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
