"""Conservative optional spelling verification for recurring proper terms."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from studio.core.workspace import published

_PAIRS = (
    ("д", "т"),
    ("б", "п"),
    ("в", "ф"),
    ("г", "к"),
    ("ж", "ш"),
    ("з", "с"),
    ("е", "э"),
    ("и", "ы"),
    ("о", "а"),
    ("ш", "щ"),
    ("ъ", "ь"),
)
_CAPITAL = re.compile(r"(?<![.!?…\n]\s)(?<!^)\b([А-ЯЁA-Z][а-яёa-z]{3,})\b", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class TermFix:
    wrong: str
    right: str
    occurrences: int
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrong": self.wrong,
            "right": self.right,
            "occurrences": self.occurrences,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class SuspectTerm:
    term: str
    occurrences: int
    context: str = ""


class TermVerifier(Protocol):
    def hits(self, term: str, context: str = "") -> int | None: ...


class WikiTermVerifier:
    """Keyless MediaWiki verifier; failed requests never become zero hits."""

    def __init__(self, *, pause: float = 1.0, timeout: float = 10.0) -> None:
        self.pause = pause
        self.timeout = timeout

    def hits(self, term: str, context: str = "") -> int | None:
        del context
        for site in ("ru.wiktionary.org", "ru.wikipedia.org"):
            query = urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": 1,
                    "format": "json",
                }
            )
            request = urllib.request.Request(
                f"https://{site}/w/api.php?{query}",
                headers={"User-Agent": "BormoStudio/0.1"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    payload = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                return None
            finally:
                if self.pause:
                    time.sleep(self.pause)
            count = int(payload.get("query", {}).get("searchinfo", {}).get("totalhits", 0))
            if count:
                return count
        return 0


def find_suspect_terms(text: str, *, min_occurrences: int = 2) -> list[SuspectTerm]:
    groups: dict[str, list[str]] = {}
    for word in _CAPITAL.findall(text or ""):
        groups.setdefault(word.lower()[:6], []).append(word)
    suspects = [
        SuspectTerm(min(forms, key=len), len(forms))
        for forms in groups.values()
        if len(forms) >= min_occurrences and min(map(len, forms)) >= 6
    ]
    return sorted(suspects, key=lambda item: -item.occurrences)


def spelling_variants(term: str, *, max_variants: int = 24) -> list[str]:
    swaps = {left: right for left, right in _PAIRS} | {right: left for left, right in _PAIRS}
    value = term.lower()
    variants: list[str] = []
    for index in range(len(value) - 1, 0, -1):
        replacement = swaps.get(value[index])
        if replacement:
            candidate = value[:index] + replacement + value[index + 1 :]
            variants.append(
                candidate[:1].upper() + candidate[1:] if term[:1].isupper() else candidate
            )
        if len(variants) >= max_variants:
            break
    return list(dict.fromkeys(variants))


def verify_terms(
    terms: Sequence[SuspectTerm],
    verifier: TermVerifier,
    *,
    cache: dict[str, int] | None = None,
    unknown_hits: int = 1,
    confident_hits: int = 5,
) -> list[TermFix]:
    values = cache if cache is not None else {}
    fixes: list[TermFix] = []
    for term in terms:
        original = _hits(values, verifier, term.term, term.context)
        if original is None or original >= unknown_hits:
            continue
        choices = [
            (variant, _hits(values, verifier, variant, term.context))
            for variant in spelling_variants(term.term)
        ]
        known = [
            (variant, hits)
            for variant, hits in choices
            if hits is not None and hits >= confident_hits
        ]
        if known:
            right, hits = max(known, key=lambda item: item[1])
            fixes.append(
                TermFix(
                    term.term, right, term.occurrences, f"{term.term}: 0 hits, {right}: {hits} hits"
                )
            )
    return fixes


def apply_term_fixes(text: str, fixes: Iterable[TermFix]) -> tuple[str, int]:
    result, count = text, 0
    for fix in fixes:
        pattern = re.compile(r"\b(" + re.escape(fix.wrong) + r")([а-яёa-z]*)\b", re.IGNORECASE)

        # A capital the fix adds on purpose (стекловерфлею → Stack Overflow) is
        # kept; otherwise the replacement follows the case of what was heard.
        deliberate = fix.wrong[:1].islower() and fix.right[:1].isupper()

        def replace(match: re.Match[str], fix: TermFix = fix, deliberate: bool = deliberate) -> str:
            head, tail = match.group(1), match.group(2)
            if deliberate or head[:1].isupper():
                return fix.right + tail
            return fix.right.lower() + tail

        result, current = pattern.subn(replace, result)
        count += current
    return result, count


def load_cache(path: Path | None) -> dict[str, int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}
    return (
        {
            str(key): item
            for key, item in value.items()
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0
        }
        if isinstance(value, Mapping)
        else {}
    )


def save_cache(path: Path | None, cache: Mapping[str, int]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with published(path) as temporary:
        temporary.write_text(
            json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _hits(cache: dict[str, int], verifier: TermVerifier, term: str, context: str) -> int | None:
    key = f"{context}|{term}".lower().strip("|")
    if key not in cache:
        found = verifier.hits(term, context)
        if found is None:
            return None
        cache[key] = found
    return cache[key]
