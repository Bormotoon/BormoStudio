"""Quotas, review payloads and gates of Forge's staged moment selection.

Forge read these from ``processing.*``; Studio keeps them as flat ``reels.*``
settings. The arithmetic is Forge's, so the same transcript and the same LLM
answers select the same moments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from studio.core.transcript_index import TranscriptIndex
from studio.reels.analysis.chunking import estimate_tokens
from studio.reels.analysis.contracts import MomentRecord
from studio.reels.analysis.ranking import ranking_value, topic_similarity
from studio.reels.analysis.scoring import clip_type_target_bounds
from studio.reels.analysis.validation import filter_nonoverlapping_outputs

# Clip types, their count setting and their length setting.
CLIP_TYPES = (
    ("story", "story_count", "story_max_s", 15),
    ("reel", "reel_count", "reel_max_s", 60),
    ("long_reel", "long_reel_count", "long_reel_max_s", 180),
    ("highlight", "highlight_count", None, None),
)


def requested_quotas(reels: Mapping[str, Any]) -> dict[str, int]:
    """Clip counts per type; ``max_candidates`` stands in for an empty reel count."""
    quotas = {bucket: max(0, int(reels.get(count, 0) or 0)) for bucket, count, *_ in CLIP_TYPES}
    if quotas["reel"] <= 0:
        quotas["reel"] = int(reels.get("max_candidates", 4))
    return quotas


def scale_quotas_to_duration(
    quotas: Mapping[str, int], *, duration_s: float, clips_per_hour: float
) -> dict[str, int]:
    """``clips_per_hour`` clips per hour of runtime, in the configured type mix.

    Largest-remainder apportionment, so the counts sum exactly to the target.
    """
    if clips_per_hour <= 0 or duration_s <= 0:
        return dict(quotas)
    base = {key: max(0, int(value)) for key, value in quotas.items()}
    base_total = sum(base.values())
    target_total = max(1, round(duration_s / 3600.0 * clips_per_hour))
    if base_total <= 0:
        return {**base, "reel": target_total}
    shares = {key: value / base_total * target_total for key, value in base.items()}
    scaled = {key: int(share) for key, share in shares.items()}
    shortfall = target_total - sum(scaled.values())
    for key, _remainder in sorted(
        ((key, shares[key] - scaled[key]) for key in shares), key=lambda item: -item[1]
    )[:shortfall]:
        scaled[key] += 1
    return scaled


def build_requirements_text(
    reels: Mapping[str, Any], quotas: Mapping[str, int], *, include_counts: bool = True
) -> str:
    """The clip ask told to the model; the scout gets lengths only.

    Counts come from the duration-scaled quotas, so the prompt and the
    selector enforce the same numbers.
    """

    def clips(count: int) -> str:
        return f"{count} clips " if include_counts else ""

    def limit(key: str, default: int) -> int:
        try:
            return int(reels.get(key, default))
        except (TypeError, ValueError):
            return default

    parts: list[str] = []
    if quotas.get("story", 0) > 0:
        parts.append(f"Stories: {clips(quotas['story'])}up to {limit('story_max_s', 15)}s")
    if quotas.get("reel", 0) > 0:
        parts.append(f"Reels: {clips(quotas['reel'])}up to {limit('reel_max_s', 60)}s")
    if quotas.get("long_reel", 0) > 0:
        parts.append(
            f"Long reels: {clips(quotas['long_reel'])}up to {limit('long_reel_max_s', 180)}s"
        )
    if quotas.get("highlight", 0) > 0:
        parts.append(
            f"Highlights: {quotas['highlight']} moments" if include_counts
            else "Highlights: 10-30s"
        )
    if not parts:
        total = sum(quotas.values()) if quotas else 4
        parts.append(
            f"Reels: {clips(max(1, total))}of {reels.get('target_min_s', 30)}-"
            f"{reels.get('target_max_s', 60)}s"
        )
    return "\n".join(parts)


def target_candidate_range(chunk_seconds: float) -> str:
    """About 4-8 scout candidates per ten minutes of chunk."""
    minutes = max(0.0, float(chunk_seconds)) / 60.0
    low = max(1, round(minutes * 0.4))
    high = max(low + 2, round(minutes * 0.8))
    return f"{low}-{high}"


def build_judge_payload(
    records: Sequence[MomentRecord],
    index: TranscriptIndex,
    *,
    max_candidates: int = 14,
    head_seconds: float = 15.0,
    tail_seconds: float = 5.0,
    max_excerpt_chars: int = 260,
) -> list[dict[str, Any]]:
    """What the judge sees: the real opening and closing words of each clip."""
    ordered = sorted(records, key=ranking_value, reverse=True)[: max(1, int(max_candidates))]
    payload: list[dict[str, Any]] = []
    for record in ordered:
        item: dict[str, Any] = {
            "candidate_id": record.candidate_id,
            "start": round(record.start, 1),
            "end": round(record.end, 1),
            "duration": round(record.end - record.start, 1),
            "quote": record.quote,
            "evidence": record.why,
            "score": record.score,
        }
        if record.title:
            item["title"] = record.title
        if record.reason_codes:
            item["reason_codes"] = list(record.reason_codes)
        if record.speaker:
            item["speaker"] = record.speaker
        if record.quote_match_ratio is not None:
            item["quote_match_ratio"] = record.quote_match_ratio
        if index:
            head = index.text_between(record.start, min(record.start + head_seconds, record.end),
                                      max_chars=max_excerpt_chars)
            tail = index.text_between(max(record.end - tail_seconds, record.start), record.end,
                                      max_chars=max_excerpt_chars // 2)
            if head:
                item["excerpt_head"] = head
            if tail:
                item["excerpt_tail"] = tail
        payload.append(item)
    return payload


def guard_stage_output(
    outputs: Sequence[MomentRecord], inputs: Sequence[MomentRecord], *, enabled: bool
) -> tuple[list[MomentRecord], int]:
    """Drop records a filtering stage invented; the input when nothing is traceable.

    Returns the records and how many were dropped.
    """
    if not enabled or not outputs:
        return list(outputs), 0
    kept = filter_nonoverlapping_outputs(outputs, inputs)
    if len(kept) == len(outputs):
        return kept, 0
    if not kept:
        return list(inputs), len(outputs)
    return kept, len(outputs) - len(kept)


def quality_filters(reels: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(reels.get(name, 0) or 0) for key, name in (
        ("min_score", "min_score"), ("min_duration", "min_duration_s"),
        ("max_duration", "max_duration_s"))}


def split_by_quality_filters(
    records: Sequence[MomentRecord], filters: Mapping[str, float]
) -> tuple[list[MomentRecord], list[tuple[MomentRecord, str]]]:
    kept: list[MomentRecord] = []
    rejected: list[tuple[MomentRecord, str]] = []
    min_score = filters.get("min_score", 0.0)
    min_duration = filters.get("min_duration", 0.0)
    max_duration = filters.get("max_duration", 0.0)
    for record in records:
        duration = record.end - record.start
        if min_score > 0 and record.score < min_score:
            rejected.append((record, "below_min_score"))
        elif min_duration > 0 and duration < min_duration:
            rejected.append((record, "shorter_than_min_duration"))
        elif 0 < max_duration < duration:
            rejected.append((record, "longer_than_max_duration"))
        else:
            kept.append(record)
    return kept, rejected


def quality_reasons(moment: Mapping[str, Any], filters: Mapping[str, float]) -> list[str]:
    """Why a moment fails the quality filters at render time (Forge's cut stage)."""
    try:
        score = float(moment.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    duration = float(moment["end"]) - float(moment["start"])
    reasons = []
    if filters.get("min_score", 0.0) > 0 and score < filters["min_score"]:
        reasons.append(f"score {score:.1f} < {filters['min_score']:.1f}")
    if filters.get("min_duration", 0.0) > 0 and duration < filters["min_duration"]:
        reasons.append(f"duration {duration:.0f}s < {filters['min_duration']:.0f}s")
    if 0 < filters.get("max_duration", 0.0) < duration:
        reasons.append(f"duration {duration:.0f}s > {filters['max_duration']:.0f}s")
    return reasons


def unreachable_buckets(quotas: Mapping[str, int], filters: Mapping[str, float]) -> list[str]:
    """Warnings for clip types whose lengths never pass the duration filters."""
    min_duration = filters.get("min_duration", 0.0)
    max_duration = filters.get("max_duration", 0.0)
    warnings = []
    for bucket, count in quotas.items():
        if int(count) <= 0:
            continue
        low, high = clip_type_target_bounds(bucket)
        if (min_duration > 0 and high < min_duration) or (0 < max_duration < low):
            warnings.append(
                f"clip type {bucket} ({low:g}-{high:g} s) conflicts with the duration filters "
                f"({min_duration:g}-{max_duration:g} s): its {int(count)} slot(s) go to "
                "other types")
    return warnings


def rejection_rows(
    records: Sequence[MomentRecord], *, stage: str, reason: str
) -> list[dict[str, Any]]:
    return [{**record.to_dict(), "rejected_at": stage, "rejection_reason": reason}
            for record in records]


def percentile(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))], 3)


def rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def topic_diversity(records: Sequence[MomentRecord]) -> float | None:
    """1 - mean pairwise topic similarity of the final set (1 = all distinct)."""
    pairs = [topic_similarity(first, second)
             for offset, first in enumerate(records) for second in records[offset + 1:]]
    return round(1.0 - sum(pairs) / len(pairs), 4) if pairs else None


# Scout prompt tokens held back for the episode context and requirements.
_SCOUT_PROMPT_RESERVE_TOKENS = 768
# Below this many transcript tokens per chunk the scout sees too little.
_MIN_CHUNK_TOKENS = 1000


def fit_scout_chunk_chars(
    texts: Sequence[str], *, max_chars: int, prompt: str, ctx_size: int, parallel: int,
    n_predict: int,
) -> int | None:
    """A smaller ``max_chars`` when the biggest scout prompt would overflow a slot.

    llama.cpp left-truncates an overflowing prompt silently. None when the
    chunks already fit or the context size is unknown.
    """
    if ctx_size <= 0 or not texts:
        return None
    slot_ctx = ctx_size // max(1, parallel)
    sample = max(texts, key=len)
    template_tokens = estimate_tokens(prompt) + _SCOUT_PROMPT_RESERVE_TOKENS
    if template_tokens + estimate_tokens(sample) + n_predict <= slot_ctx:
        return None
    input_tokens = slot_ctx - n_predict - template_tokens
    if input_tokens < _MIN_CHUNK_TOKENS:
        return None
    chars_per_token = len(sample) / max(1, estimate_tokens(sample))
    fitted = min(int(max_chars), int(input_tokens * chars_per_token * 0.95))
    return fitted if fitted < int(max_chars) else None
