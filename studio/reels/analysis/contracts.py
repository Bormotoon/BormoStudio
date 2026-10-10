"""Typed contracts for local reels candidate analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class AnalysisChunkUnit:
    source_segment_index: int
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass(frozen=True, slots=True)
class AnalysisChunk:
    chunk_id: str
    start: float
    end: float
    text: str
    speaker_set: tuple[str, ...] = ()
    sentence_count: int = 0
    overlap_left: bool = False
    overlap_right: bool = False
    source_segment_ids: tuple[int, ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker_set": list(self.speaker_set),
            "sentence_count": self.sentence_count,
            "overlap_left": self.overlap_left,
            "overlap_right": self.overlap_right,
            "source_segment_ids": list(self.source_segment_ids),
        }


@dataclass(frozen=True, slots=True)
class MomentRecord:
    start: float
    end: float
    title: str
    quote: str
    why: str
    score: float
    clip_type: str = "reel"
    hook: str = ""
    caption: str = ""
    hashtags: tuple[str, ...] = ()
    candidate_id: str = ""
    merged_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    priority: float | None = None
    judge_score: float | None = None
    hook_score: float | None = None
    completeness_score: float | None = None
    speaker_focus: float | None = None
    subtitle_readability_score: float | None = None
    duration_fit_score: float | None = None
    quote_match_ratio: float | None = None
    quote_match_method: str = ""
    quote_start: float | None = None
    quote_end: float | None = None
    speech_rate_wps: float | None = None
    audio_energy_db: float | None = None
    audio_silence_ratio: float | None = None
    replay_score: float | None = None
    selection_stage: str = ""
    source_chunk_ids: tuple[str, ...] = ()
    speaker: str = ""
    speaker_confidence: float | None = None
    derived_fields: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "title": self.title,
            "quote": self.quote,
            "why": self.why,
            "score": self.score,
            "clip_type": self.clip_type,
            "hook": self.hook,
            "caption": self.caption,
            "hashtags": list(self.hashtags),
        }
        optional = {
            "candidate_id": self.candidate_id,
            "merged_ids": list(self.merged_ids),
            "reason_codes": list(self.reason_codes),
            "priority": self.priority,
            "judge_score": self.judge_score,
            "hook_score": self.hook_score,
            "completeness_score": self.completeness_score,
            "speaker_focus": self.speaker_focus,
            "subtitle_readability_score": self.subtitle_readability_score,
            "duration_fit_score": self.duration_fit_score,
            "quote_match_ratio": self.quote_match_ratio,
            "quote_match_method": self.quote_match_method,
            "quote_start": _rounded(self.quote_start),
            "quote_end": _rounded(self.quote_end),
            "speech_rate_wps": self.speech_rate_wps,
            "audio_energy_db": self.audio_energy_db,
            "audio_silence_ratio": self.audio_silence_ratio,
            "replay_score": self.replay_score,
            "selection_stage": self.selection_stage,
            "source_chunk_ids": list(self.source_chunk_ids),
            "speaker": self.speaker,
            "speaker_confidence": self.speaker_confidence,
            "derived_fields": list(self.derived_fields),
        }
        data.update(
            {key: value for key, value in optional.items() if value not in (None, "", [], ())}
        )
        data.update(
            {
                key: value
                for key, value in self.extra.items()
                if key not in data and value is not None
            }
        )
        return data


_KNOWN_KEYS = frozenset(item.name for item in fields(MomentRecord)) - {"extra"}
_DROPPED_KEYS = frozenset({"crop_confidence", "evidence"})


def coerce_moment_record(raw: Mapping[str, Any]) -> MomentRecord | None:
    try:
        start, end = float(raw.get("start", 0)), float(raw.get("end", 0))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    why = str(raw.get("why") or raw.get("evidence") or "").strip()
    if not raw.get("title") and not raw.get("quote") and not why:
        return None
    return MomentRecord(
        start=start,
        end=end,
        title=str(raw.get("title") or "").strip(),
        quote=str(raw.get("quote") or "").strip(),
        why=why,
        score=_number(raw.get("score")) or 0.0,
        clip_type=str(raw.get("clip_type") or "reel").strip() or "reel",
        hook=str(raw.get("hook") or "").strip(),
        caption=str(raw.get("caption") or "").strip(),
        hashtags=_strings(raw.get("hashtags")),
        candidate_id=str(raw.get("candidate_id") or "").strip(),
        merged_ids=_strings(raw.get("merged_ids")),
        reason_codes=_strings(raw.get("reason_codes")),
        priority=_number(raw.get("priority")),
        judge_score=_number(raw.get("judge_score")),
        hook_score=_number(raw.get("hook_score")),
        completeness_score=_number(raw.get("completeness_score")),
        speaker_focus=_number(raw.get("speaker_focus")),
        subtitle_readability_score=_number(raw.get("subtitle_readability_score")),
        duration_fit_score=_number(raw.get("duration_fit_score")),
        quote_match_ratio=_number(raw.get("quote_match_ratio")),
        quote_match_method=str(raw.get("quote_match_method") or "").strip(),
        quote_start=_number(raw.get("quote_start")),
        quote_end=_number(raw.get("quote_end")),
        speech_rate_wps=_number(raw.get("speech_rate_wps")),
        audio_energy_db=_number(raw.get("audio_energy_db")),
        audio_silence_ratio=_number(raw.get("audio_silence_ratio")),
        replay_score=_number(raw.get("replay_score")),
        selection_stage=str(raw.get("selection_stage") or "").strip(),
        source_chunk_ids=_strings(raw.get("source_chunk_ids")),
        speaker=str(raw.get("speaker") or "").strip(),
        speaker_confidence=_number(raw.get("speaker_confidence")),
        derived_fields=_strings(raw.get("derived_fields")),
        extra={key: value for key, value in raw.items() if key not in _KNOWN_KEYS | _DROPPED_KEYS},
    )


def has_quote_evidence(raw: Mapping[str, Any] | MomentRecord) -> bool:
    quote = raw.quote if isinstance(raw, MomentRecord) else raw.get("quote", "")
    return len(str(quote or "").split()) >= 2


def replace_record(record: MomentRecord, **changes: Any) -> MomentRecord:
    return coerce_moment_record({**record.to_dict(), **changes}) or record


def _strings(value: Any) -> tuple[str, ...]:
    return (
        tuple(str(item).strip() for item in value if str(item).strip())
        if isinstance(value, (list, tuple))
        else ()
    )


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _rounded(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None
