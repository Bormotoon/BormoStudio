"""Preserved planning-to-rendering facade for an already aligned camera clip."""

from dataclasses import replace
from pathlib import Path

from studio.core.timeline import TimeDomain
from studio.core.transcript import Transcript
from studio.stages.sync_acoustic import refine_piece_boundaries
from studio.stages.sync_acoustic_settings import AcousticSettings
from studio.stages.sync_geometry import AlignmentMap, PieceSettings
from studio.stages.sync_match_settings import MatchSettings
from studio.stages.sync_matcher import align, evaluate_alignment
from studio.stages.sync_models import Segment as MatchSegment
from studio.stages.sync_models import Transcript as MatchTranscript
from studio.stages.sync_models import Word as MatchWord
from studio.stages.sync_pieces import clip_pieces, recorder_word_gaps
from studio.stages.sync_plan import RenderedAudioPlan
from studio.stages.sync_refine import RefineResult, RefineSettings, refine_plan
from studio.stages.sync_render import PauseDuck, render_audio_plan


def matching_transcript(transcript: Transcript) -> MatchTranscript:
    """Copy immutable public words before the matcher assigns normalized tokens."""
    if transcript.time_domain is not TimeDomain.FILE:
        raise ValueError("alignment requires file-domain transcripts")
    return MatchTranscript(
        transcript.source_audio, transcript.language, transcript.duration,
        [MatchSegment(segment.start, segment.end, [
            MatchWord(word.text, word.start, word.end, word.probability)
            for word in segment.words
        ]) for segment in transcript.segments],
    )


def synchronize_transcripts(
    camera: Transcript, recorder: Transcript, source: Path, output: Path, *,
    match_settings: MatchSettings, piece_settings: PieceSettings,
    strategy: int, source_asset_id: str, target_asset_id: str,
    sample_rate: int = 48000, channels: int = 1, codec: str = "pcm_s24le",
) -> RenderedAudioPlan:
    """Align public transcripts, reject unsupported fits, and render one clip."""
    alignment = align(matching_transcript(camera), matching_transcript(recorder), match_settings)
    verdict = evaluate_alignment(alignment, camera.duration, match_settings)
    if not verdict.accepted:
        raise ValueError(f"unreliable transcript alignment: {verdict.reason_text}")
    return render_aligned_clip(
        alignment, piece_settings, source, output,
        clip_duration_s=camera.duration, recorder_duration_s=recorder.duration,
        recorder_words=[(word.start, word.end) for word in recorder.words],
        strategy=strategy, source_asset_id=source_asset_id, target_asset_id=target_asset_id,
        sample_rate=sample_rate, channels=channels, codec=codec,
    )


def render_aligned_clip(
    alignment: AlignmentMap, settings: PieceSettings, source: Path, output: Path,
    *, clip_duration_s: float, recorder_duration_s: float,
    recorder_words: list[tuple[float, float]], strategy: int,
    source_asset_id: str, target_asset_id: str,
    sample_rate: int = 48000, channels: int = 1, codec: str = "pcm_s24le",
    fade_ms: int = 10, stretch_method: str = "auto",
    camera_audio: Path | None = None,
    refine_audio: Path | None = None,
    refine: RefineSettings | None = None,
    label: str = "",
    duck: PauseDuck | None = None,
) -> RenderedAudioPlan:
    """Plan pieces, refine them acoustically against ``refine_audio`` (the camera's
    own sound) when ``refine`` is given, apply Boundary Flex with ``camera_audio``,
    then render."""
    if strategy not in {1, 2, 3}:
        raise ValueError("synchronization strategy must be 1, 2, or 3")
    lead, pieces = clip_pieces(
        alignment, clip_duration_s, recorder_duration_s, strategy, settings,
        rec_word_gaps=recorder_word_gaps(recorder_words), rec_words=recorder_words,
    )
    refined: RefineResult | None = None
    if refine is not None and refine_audio is not None and pieces:
        try:
            refined = refine_plan(
                lead, pieces, refine_audio, source, clip_duration_s=clip_duration_s,
                recorder_duration_s=recorder_duration_s, recorder_words=recorder_words,
                settings=refine, label=label or target_asset_id,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            refined = RefineResult(lead, pieces, False,
                                   {"accepted": False, "reason": f"not measured: {exc}"})
        lead, pieces = refined.lead, refined.pieces
    if camera_audio is not None:
        lead, pieces = refine_piece_boundaries(
            pieces, lead, camera_audio, source, clip_duration_s, recorder_duration_s,
            AcousticSettings(),
        )
    if any(start < 0 or start + duration > recorder_duration_s + 1e-6
           for start, duration, _factor in pieces):
        raise ValueError("piece plan reads outside the recorder")
    plan = render_audio_plan(
        source, output, pieces, lead_silence_s=lead, duration_s=clip_duration_s,
        map_id=f"warp-{source_asset_id}-{target_asset_id}",
        source_asset_id=source_asset_id, target_asset_id=target_asset_id, strategy=strategy,
        sample_rate=sample_rate, channels=channels, codec=codec, fade_ms=fade_ms,
        stretch_method=stretch_method, duck=duck,
    )
    if refined is None:
        return plan
    evidence = dict(plan.warp.evidence or {})
    if refined.accepted:
        evidence.update(steps=float(len(refined.report.get("steps", []))),
                        refine_p90_ms=float(refined.report.get("p90_after_ms", 0.0)))
    return replace(plan, warp=replace(plan.warp, evidence=evidence or None),
                   refine=refined.report, warnings=tuple(refined.warnings))
