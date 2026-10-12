"""JSON schemas of the reels LLM answers (Forge), sent as constrained sampling.

JSON schemas for LLM responses, used as llama.cpp grammars.

llama.cpp converts ``json_schema`` into a GBNF grammar and samples strictly
against it, so the schema has to stay inside the supported subset: plain
types, ``properties``, ``required`` and ``items``. No ``$ref``, ``oneOf`` or
similar — the server answers 400 on those.
"""

from __future__ import annotations

from typing import Any

# Schema of a single moment in the legacy full-record format. Kept for
# backward compatibility: custom prompts and cloud providers may still answer
# with a moments list.
MOMENT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "clip_type": {"type": "string"},
        "title": {"type": "string"},
        "quote": {"type": "string"},
        "why": {"type": "string"},
        "score": {"type": "number"},
        "hook": {"type": "string"},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start", "end", "quote", "score"],
}

MOMENTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "moments": {"type": "array", "items": MOMENT_ITEM_SCHEMA},
    },
    "required": ["moments"],
}

# The scout returns evidence and a rough rating only: the interval, a
# verbatim quote, a one-line justification and reason codes. Titles,
# captions and hashtags do not belong here — they burned n_predict and cost
# recall.
SCOUT_CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "quote": {"type": "string"},
        "evidence": {"type": "string"},
        "reason_codes": {"type": "array", "items": {"type": "string"}},
        "score": {"type": "number"},
    },
    "required": ["start", "end", "quote", "evidence", "score"],
}

SCOUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": SCOUT_CANDIDATE_SCHEMA},
    },
    "required": ["candidates"],
}

# Cleanup does not echo records back; it returns decisions by candidate_id.
CLEANUP_DECISIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "merge_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["candidate_id", "keep"],
            },
        },
    },
    "required": ["decisions"],
}

# The judge rates candidates by id and writes metadata for them only.
# quote, start and end are out of its reach by construction.
JUDGE_REVIEWS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "keep": {"type": "boolean"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "why": {"type": "string"},
                    "reason_codes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["candidate_id", "keep", "score"],
            },
        },
    },
    "required": ["reviews"],
}

# Episode-overview schema. Kept separate from moments: the grammar is
# strict, so a provider carrying the moments schema cannot answer
# {"summary": ...} at all.
EPISODE_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "string"},
        "speakers": {"type": "array", "items": {"type": "string"}},
        "context_limits": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}

# The permissive "any object" schema — the original behaviour and the
# downgrade path for llama.cpp builds that reject the full one.
ANY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}


def _flat_list_schema(key: str, fields: dict[str, str], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            key: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {name: {"type": kind} for name, kind in fields.items()},
                    "required": required,
                },
            },
        },
        "required": [key],
    }


# Simplified schemas for llama.cpp builds that reject the full one. No
# nested arrays and a minimum of fields, but still with the evidence fields
# required — instead of dropping straight to "any object".
FALLBACK_SCHEMAS: dict[str, dict[str, Any]] = {
    "scout": _flat_list_schema(
        "candidates",
        {"start": "number", "end": "number", "quote": "string", "score": "number"},
        ["start", "end", "quote"],
    ),
    "cleanup": _flat_list_schema(
        "decisions",
        {"candidate_id": "string", "keep": "boolean"},
        ["candidate_id", "keep"],
    ),
    "judge": _flat_list_schema(
        "reviews",
        {"candidate_id": "string", "keep": "boolean", "score": "number", "title": "string"},
        ["candidate_id", "keep", "score"],
    ),
    "moments": _flat_list_schema(
        "moments",
        {"start": "number", "end": "number", "quote": "string", "score": "number"},
        ["start", "end", "quote"],
    ),
    "context": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}
