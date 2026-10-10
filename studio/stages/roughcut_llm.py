"""Optional LLM rough-cut detectors with transcript-evidence guardrails.

The model only names phrase numbers and quotes text. Every time comes from the
transcript: a quote must match a contiguous run of words in the referenced
phrase, and the cut snaps to those word boundaries. Anything that cannot be
proven against the words is rejected and reported, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

from studio.core.transcript import Transcript, Word
from studio.stages.retake_models import RetakeGroup

DETECTORS = ("start", "slips", "fillers", "delivery")
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

_TASKS = {
    "ru": {
        "start": "start — где на самом деле начинается разговор: номер первой фразы "
                 "по существу после проверки звука, отсчёта и подготовки. null, если "
                 "разговор начинается сразу.",
        "slips": "slips — оговорки: неверное слово или начало фразы, которое спикер "
                 "тут же поправил. quote — только ошибочный кусок, без поправки.",
        "fillers": "fillers — паразиты по смыслу: «как бы», «в общем», «то есть», "
                   "которые ничего не добавляют именно в этом месте.",
    },
    "en": {
        "start": "start — where the conversation really begins: the number of the first "
                 "substantive phrase after sound checks, countdowns and setup. null if "
                 "the conversation starts right away.",
        "slips": "slips — slips of the tongue: a wrong word or false start the speaker "
                 "immediately corrected. quote is only the wrong part, not the fix.",
        "fillers": "fillers — semantic fillers: \"like\", \"you know\", \"I mean\" that add "
                   "nothing at this exact spot.",
    },
}


@dataclass(frozen=True, slots=True)
class Phrase:
    id: int
    words: tuple[Word, ...]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


@dataclass(frozen=True, slots=True)
class Finding:
    detector: str
    start: float
    end: float
    quote: str
    reason: str
    confidence: float


@dataclass(slots=True)
class LlmResult:
    findings: list[Finding] = field(default_factory=list)
    delivery: dict[int, tuple[int, str, float]] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    requests: int = 0

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "accepted": [
                {"detector": item.detector, "start": round(item.start, 3),
                 "end": round(item.end, 3), "quote": item.quote, "reason": item.reason,
                 "confidence": round(item.confidence, 3)}
                for item in self.findings
            ],
            "delivery": [
                {"group": group, "take": take, "reason": reason,
                 "confidence": round(confidence, 3)}
                for group, (take, reason, confidence) in sorted(self.delivery.items())
            ],
            "rejected": self.rejected,
            "failures": self.failures,
            "requests": self.requests,
        }


@dataclass(frozen=True, slots=True)
class LlmOptions:
    detectors: tuple[str, ...] = DETECTORS
    language: str = "ru"
    min_confidence: float = 0.6
    max_cut_s: float = 8.0
    max_start_s: float = 600.0
    chunk_chars: int = 6000


def build_phrases(transcript: Transcript) -> list[Phrase]:
    return [
        Phrase(index, segment.words)
        for index, segment in enumerate(
            item for item in transcript.segments if item.words
        )
    ]


def chunk_phrases(phrases: list[Phrase], max_chars: int) -> list[list[Phrase]]:
    chunks: list[list[Phrase]] = []
    current: list[Phrase] = []
    size = 0
    for phrase in phrases:
        line = len(phrase.text) + 12
        if current and size + line > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(phrase)
        size += line
    if current:
        chunks.append(current)
    return chunks


def prompt_text(name: str, language: str) -> str:
    root = files("studio.resources").joinpath("prompts")
    candidate = root.joinpath(language, f"{name}.txt")
    if not candidate.is_file():
        candidate = root.joinpath("ru", f"{name}.txt")
    return candidate.read_text(encoding="utf-8")


def chunk_prompt(chunk: list[Phrase], options: LlmOptions, *, first: bool) -> str:
    tasks = _TASKS.get(options.language, _TASKS["ru"])
    enabled = [
        tasks[name] for name in ("start", "slips", "fillers")
        if name in options.detectors and (name != "start" or first)
    ]
    lines = "\n".join(
        f"[{phrase.id}] {phrase.start:.1f}s: {phrase.text}" for phrase in chunk
    )
    return (prompt_text("roughcut_chunk", options.language)
            .replace("{tasks}", "\n".join(f"- {item}" for item in enabled))
            .replace("{phrases}", lines))


def delivery_prompt(groups: list[RetakeGroup], language: str) -> str:
    blocks = []
    for index, group in enumerate(groups):
        takes = "\n".join(
            f"  take {number}: {take.text}" for number, take in enumerate(group.takes)
        )
        blocks.append(f"group {index}:\n{takes}")
    return prompt_text("roughcut_delivery", language).replace("{groups}", "\n".join(blocks))


def run_detectors(
    transcript: Transcript,
    groups: list[RetakeGroup],
    request: Callable[[str], Any],
    options: LlmOptions,
) -> LlmResult:
    """Run the enabled detectors; ``request`` returns a parsed JSON payload."""
    result = LlmResult()
    phrases = build_phrases(transcript)
    by_id = {phrase.id: phrase for phrase in phrases}
    chunk_detectors = {"start", "slips", "fillers"} & set(options.detectors)
    if phrases and chunk_detectors:
        for number, chunk in enumerate(chunk_phrases(phrases, options.chunk_chars)):
            if not (chunk_detectors - {"start"}) and number:
                break
            result.requests += 1
            try:
                payload = request(chunk_prompt(chunk, options, first=number == 0))
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                result.failures.append(f"chunk {number}: {exc}")
                continue
            if not isinstance(payload, dict):
                result.failures.append(f"chunk {number}: response is not an object")
                continue
            allowed = {phrase.id for phrase in chunk}
            if number == 0 and "start" in options.detectors:
                _accept_start(payload.get("start"), phrases, by_id, allowed, options, result)
            for detector in ("slips", "fillers"):
                if detector not in options.detectors:
                    continue
                items = payload.get(detector, [])
                if not isinstance(items, list):
                    result.rejected.append({"detector": detector, "why": "not a list"})
                    continue
                for item in items:
                    _accept_quote(detector, item, by_id, allowed, options, result)
    if groups and "delivery" in options.detectors:
        result.requests += 1
        try:
            payload = request(delivery_prompt(groups, options.language))
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            result.failures.append(f"delivery: {exc}")
        else:
            _accept_delivery(payload, groups, options, result)
    return result


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def _reject(result: LlmResult, detector: str, item: Any, why: str) -> None:
    result.rejected.append({"detector": detector, "why": why, "item": item})


def _accept_start(
    item: Any, phrases: list[Phrase], by_id: dict[int, Phrase], allowed: set[int],
    options: LlmOptions, result: LlmResult,
) -> None:
    if item is None:
        return
    if not isinstance(item, dict) or isinstance(item.get("phrase"), bool) or not isinstance(
        item.get("phrase"), int
    ):
        _reject(result, "start", item, "phrase is not an integer")
        return
    phrase_id = item["phrase"]
    if phrase_id not in allowed:
        _reject(result, "start", item, "unknown phrase")
        return
    if phrase_id == phrases[0].id:
        return
    start, end = phrases[0].start, by_id[phrase_id].start
    if end > options.max_start_s:
        _reject(result, "start", item, "start is beyond llm_max_start_s")
        return
    confidence = _confidence(item.get("confidence", 1.0))
    if confidence is None:
        _reject(result, "start", item, "invalid confidence")
        return
    if confidence < options.min_confidence:
        _reject(result, "start", item, "below llm_min_confidence")
        return
    quote = " ".join(phrase.text for phrase in phrases if phrase.id < phrase_id)
    result.findings.append(
        Finding("start", start, end, quote, str(item.get("reason", ""))[:200], confidence)
    )


def _accept_quote(
    detector: str, item: Any, by_id: dict[int, Phrase], allowed: set[int],
    options: LlmOptions, result: LlmResult,
) -> None:
    if not isinstance(item, dict):
        _reject(result, detector, item, "not an object")
        return
    phrase_id, quote = item.get("phrase"), item.get("quote")
    if isinstance(phrase_id, bool) or not isinstance(phrase_id, int) or phrase_id not in allowed:
        _reject(result, detector, item, "unknown phrase")
        return
    if not isinstance(quote, str) or not quote.strip():
        _reject(result, detector, item, "missing quote")
        return
    confidence = _confidence(item.get("confidence"))
    if confidence is None:
        _reject(result, detector, item, "invalid confidence")
        return
    if confidence < options.min_confidence:
        _reject(result, detector, item, "below llm_min_confidence")
        return
    span = locate_quote(by_id[phrase_id].words, quote)
    if span is None:
        _reject(result, detector, item, "quote not found verbatim in phrase")
        return
    words = by_id[phrase_id].words[span[0]:span[1]]
    if len(words) == len(by_id[phrase_id].words) and detector == "fillers":
        _reject(result, detector, item, "filler quote covers the whole phrase")
        return
    start, end = words[0].start, words[-1].end
    if end - start > options.max_cut_s:
        _reject(result, detector, item, "longer than llm_max_cut_s")
        return
    result.findings.append(Finding(
        detector, start, end, " ".join(word.text for word in words),
        str(item.get("reason", ""))[:200], confidence,
    ))


def _accept_delivery(
    payload: Any, groups: list[RetakeGroup], options: LlmOptions, result: LlmResult
) -> None:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list):
        result.failures.append("delivery: response has no choices list")
        return
    for item in choices:
        if not isinstance(item, dict):
            _reject(result, "delivery", item, "not an object")
            continue
        group, take = item.get("group"), item.get("take")
        if (isinstance(group, bool) or not isinstance(group, int)
                or not 0 <= group < len(groups)):
            _reject(result, "delivery", item, "unknown group")
            continue
        if (isinstance(take, bool) or not isinstance(take, int)
                or not 0 <= take < len(groups[group].takes)):
            _reject(result, "delivery", item, "unknown take")
            continue
        if group in result.delivery:
            _reject(result, "delivery", item, "duplicate group")
            continue
        confidence = _confidence(item.get("confidence"))
        if confidence is None:
            _reject(result, "delivery", item, "invalid confidence")
            continue
        if confidence < options.min_confidence:
            _reject(result, "delivery", item, "below llm_min_confidence")
            continue
        result.delivery[group] = (take, str(item.get("reason", ""))[:200], confidence)


def _tokens(text: str) -> list[str]:
    return [token.casefold().replace("ё", "е") for token in _TOKEN_RE.findall(text)]


def locate_quote(words: tuple[Word, ...], quote: str) -> tuple[int, int] | None:
    """Return the word index range whose normalized tokens equal the quote."""
    wanted = _tokens(quote)
    if not wanted:
        return None
    per_word = [_tokens(word.text) for word in words]
    for first in range(len(words)):
        if not per_word[first]:
            continue
        collected: list[str] = []
        for last in range(first, len(words)):
            collected.extend(per_word[last])
            if collected == wanted:
                return first, last + 1
            if len(collected) >= len(wanted) or collected != wanted[:len(collected)]:
                break
    return None
