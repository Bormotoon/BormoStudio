"""Publish review-program captions separately from the clean master."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studio.core.project import Project, stable_fingerprint
from studio.core.subtitles import edit_path, load_edits, to_ass
from studio.core.transcript import Segment, Transcript, to_srt
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.reel_render import _digest, burn_subtitles


class ProgramSubtitlesStage:
    id = "program_subtitles"
    title = "Review master captions"
    after: tuple[str, ...] = ("program",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ()
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        return []

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        if not settings.get("program", {}).get("burn_subtitles", False):
            return Decision.skip("program captions are disabled")
        if not project.outputs.get("program") or not project.transcripts.get("edited"):
            return Decision.blocked(
                "review master and edited transcript are required", "run program"
            )
        return Decision.run()

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        paths = [*project.outputs.get("program", []), edit_path(project.work_dir, "program")]
        if project.transcripts.get("edited"):
            paths.append(project.transcripts["edited"])
        return stable_fingerprint(
            "program-captions-v1", [(str(path), _digest(path)) for path in paths if path.is_file()]
        )

    def run(self, context: StageContext) -> StageOutput:
        transcript = Transcript.load(context.project.transcripts["edited"])
        cues, style = load_edits(context.project.work_dir, "program", transcript)
        directory = context.work_dir / "stages" / self.id
        directory.mkdir(parents=True, exist_ok=True)
        ass, srt, video = (
            directory / name for name in ("program.ass", "program.srt", "program-captioned.mp4")
        )
        ass.write_text(to_ass(cues, style), encoding="utf-8")
        local = Transcript(
            transcript.source_audio,
            transcript.language,
            transcript.duration,
            [Segment(cue.start, cue.end, text_override=cue.text) for cue in cues],
        )
        srt.write_text(to_srt(local), encoding="utf-8")
        burn_subtitles(context.project.outputs["program"][0], ass, video)
        artifacts: tuple[Path, ...] = (video, srt, ass)
        return StageOutput(
            artifacts,
            {
                "outputs": {
                    **context.project.outputs,
                    self.id: list(artifacts),
                }
            },
        )
