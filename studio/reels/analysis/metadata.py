"""Presentation metadata of final moments, migrated from Forge.

Only presentation fields (title, hook, caption, hashtags, why) are filled in
here, and every one that is filled in is listed in ``derived_fields``.
Evidence fields (``start``, ``end``, ``quote``) are never synthesized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from studio.reels.analysis.contracts import (
    MomentRecord,
    coerce_moment_record,
    has_quote_evidence,
)
from studio.reels.markdown import build_description_text, build_hashtags


class MissingEvidenceError(ValueError):
    """A moment reached metadata finalization without a transcript quote."""


def finalize_moment_metadata(
    moment: MomentRecord | Mapping[str, Any], *, require_quote: bool = True
) -> MomentRecord:
    raw = moment.to_dict() if isinstance(moment, MomentRecord) else dict(moment)
    if require_quote and not has_quote_evidence(raw):
        raise MissingEvidenceError(
            f"moment [{raw.get('start')}, {raw.get('end')}] has no transcript quote"
        )
    derived = list(raw.get("derived_fields") or [])

    def derive(key: str, value: Any) -> None:
        raw[key] = value
        if key not in derived:
            derived.append(key)

    description = build_description_text(raw)
    if not str(raw.get("title", "") or "").strip():
        derive("title", description[:60])
    if not str(raw.get("hook", "") or "").strip():
        derive("hook", raw["title"])
    if not str(raw.get("why", "") or "").strip():
        derive("why", description)
    if not str(raw.get("caption", "") or "").strip():
        derive("caption", description)
    hashtags = build_hashtags(raw, description_text=description)
    if list(hashtags[:5]) != list(raw.get("hashtags") or []):
        derive("hashtags", hashtags[:5])
    record = coerce_moment_record(
        {**raw, "derived_fields": derived,
         "selection_stage": raw.get("selection_stage") or "metadata"}
    )
    if record is None:
        raise ValueError("failed to finalize moment metadata")
    return record


def finalize_moment_list(
    moments: Sequence[MomentRecord | Mapping[str, Any]], *, require_quote: bool = True
) -> tuple[list[MomentRecord], list[MomentRecord | Mapping[str, Any]]]:
    """Finalized moments and the ones rejected for lacking evidence."""
    final: list[MomentRecord] = []
    rejected: list[MomentRecord | Mapping[str, Any]] = []
    for moment in moments:
        try:
            final.append(finalize_moment_metadata(moment, require_quote=require_quote))
        except MissingEvidenceError:
            rejected.append(moment)
    return final, rejected
