"""Local LLM text outputs with conservative transcript and article guardrails."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Any, TypedDict

from studio.core.project import Project, stable_fingerprint
from studio.core.transcript import Segment, Transcript, to_srt
from studio.core.transcript_align import realign_transcript_words
from studio.llm.providers import LlamaProvider, parse_json_response
from studio.llm.roles import role_provider
from studio.llm.session import model_identity
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class ArticleSection:
    title: str
    paragraphs: tuple[str, ...]
    start: float
    end: float

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "paragraphs": list(self.paragraphs),
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }


class TextStage:
    id = "text"
    title = "Proofread and article"
    after: tuple[str, ...] = ("timeline",)
    gpu = GpuUse.LLM

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        text = _text_settings(settings)
        if not _enabled(text):
            return []
        return [Requirement("llama", str(text["model"]), "generate local text outputs")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        text = _text_settings(settings)
        if not _enabled(text):
            return Decision.skip("text outputs are disabled")
        if "timeline" not in project.transcripts:
            return Decision.blocked("timeline transcript is missing", "run timeline")
        return Decision.run(
            {"proofread": bool(text["proofread"]), "article": bool(text["article"])}
        )

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        source = project.transcripts.get("timeline")
        content = source.read_text(encoding="utf-8") if source and source.is_file() else ""
        return stable_fingerprint(
            "text-v3",
            content,
            _text_settings(settings),
            model_identity(settings),
            settings.get("llm", {}).get("roles", {}),
            [
                _prompt(name, language)
                for name in ("proofread_default", "article_default")
                for language in ("ru", "en")
            ],
        )

    def run(self, context: StageContext) -> StageOutput:
        settings = _text_settings(context.settings)
        source = Transcript.load(context.project.transcripts["timeline"])
        provider = LlamaProvider(
            str(settings["base_url"]),
            str(settings["model"]),
            context.project.work_dir / "cache" / "llm",
        )
        if context.settings.get("llm", {}).get("managed", False):
            provider = LlamaProvider(
                provider.base_url,
                provider.model,
                provider.cache_dir,
                model_identity(context.settings),
            )
        artifacts: list[Path] = []
        outputs = {**context.project.outputs}
        transcripts = {**context.project.transcripts}
        text_dir = context.work_dir / "stages" / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        active = source
        if settings["proofread"]:
            provider = role_provider(
                context.settings,
                "text",
                "proofread",
                context.project.work_dir / "cache" / "llm",
                LlamaProvider,
            )
            active, report = proofread_transcript(active, provider, settings)
            if settings["term_check"]:
                active, term_report = check_transcript_terms(
                    active, settings, context.project.work_dir / "cache" / "terms.json"
                )
                report["term_check"] = term_report
            proofread_path = text_dir / "proofread.json"
            active.save(proofread_path)
            srt_path = proofread_path.with_suffix(".srt")
            srt_path.write_text(to_srt(active), encoding="utf-8")
            report_path = text_dir / "proofread-report.json"
            _write_json(report_path, report)
            artifacts.extend((proofread_path, srt_path, report_path))
            transcripts["proofread"] = proofread_path
        if settings["article"]:
            provider = role_provider(
                context.settings,
                "text",
                "article",
                context.project.work_dir / "cache" / "llm",
                LlamaProvider,
            )
            sections, report = build_article(active, provider, settings)
            markdown_path = text_dir / "article.md"
            markdown_path.write_text(render_article_markdown("Episode", sections), encoding="utf-8")
            report_path = text_dir / "article.json"
            _write_json(report_path, report)
            artifacts.extend((markdown_path, report_path))
        if artifacts:
            outputs["text"] = artifacts
        return StageOutput(tuple(artifacts), {"outputs": outputs, "transcripts": transcripts})


def _text_settings(settings: dict[str, Any]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "proofread": False,
        "article": False,
        "base_url": "http://127.0.0.1:8080",
        "model": "local",
        "prompt_language": "auto",
        "max_chars_chunk": 4000,
        "article_max_chars_chunk": 6000,
        "min_similarity": 0.8,
        "term_check": False,
        "term_check_network": False,
        "term_fixes": {},
        "term_max_candidates": 10,
    }
    configured = settings.get("text", {})
    if isinstance(configured, dict):
        defaults.update(configured)
    return defaults


def _enabled(settings: dict[str, Any]) -> bool:
    return bool(settings["proofread"]) or bool(settings["article"])


def proofread_transcript(
    transcript: Transcript, provider: LlamaProvider, settings: dict[str, Any]
) -> tuple[Transcript, dict[str, object]]:
    language = _prompt_language(transcript.language, str(settings["prompt_language"]))
    template = _prompt("proofread_default", language)
    segments = list(transcript.segments)
    applied = rejected = failed_batches = 0
    for indices in _batches(segments, int(settings["max_chars_chunk"])):
        payload = [{"id": index, "text": segments[index].text} for index in indices]
        try:
            response = provider.complete(
                template.replace("{glossary}", "").replace(
                    "{segments_json}", json.dumps(payload, ensure_ascii=False)
                )
            )
            corrections = _corrections(parse_json_response(response))
        except Exception:
            failed_batches += 1
            continue
        for index in indices:
            correction = corrections.get(index)
            if correction is None or correction == segments[index].text:
                continue
            if is_correction_safe(
                segments[index].text, correction, float(settings["min_similarity"])
            ):
                segments[index] = replace(segments[index], text_override=correction)
                applied += 1
            else:
                rejected += 1
    corrected = replace(transcript, segments=segments)
    corrected, words_realigned = realign_transcript_words(corrected)
    report: dict[str, object] = {
        "schema_version": 1,
        "model": settings["model"],
        "prompt_language": language,
        "segments_total": len(segments),
        "applied": applied,
        "rejected": rejected,
        "failed_batches": failed_batches,
        "words_realigned": words_realigned,
        "word_timing_text": "corrected",
    }
    corrected = replace(corrected, metadata={**transcript.metadata, "proofread": report})
    return corrected, report


def check_transcript_terms(
    transcript: Transcript, settings: dict[str, Any], cache_path: Path
) -> tuple[Transcript, dict[str, Any]]:
    from studio.stages.term_check import (
        TermFix,
        WikiTermVerifier,
        apply_term_fixes,
        find_suspect_terms,
        load_cache,
        save_cache,
        verify_terms,
    )

    fixes = [TermFix(wrong, right, 0, "manual") for wrong, right in settings["term_fixes"].items()]
    errors = []
    if settings["term_check_network"]:
        try:
            cache = load_cache(cache_path)
            terms = find_suspect_terms("\n".join(segment.text for segment in transcript.segments))[
                : int(settings["term_max_candidates"])
            ]
            fixes += verify_terms(terms, WikiTermVerifier(), cache=cache)
            save_cache(cache_path, cache)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    segments, count = [], 0
    for segment in transcript.segments:
        corrected, changes = apply_term_fixes(segment.text, fixes)
        count += changes
        segments.append(replace(segment, text_override=corrected) if changes else segment)
    result, _ = realign_transcript_words(replace(transcript, segments=segments))
    return result, {
        "network_enabled": settings["term_check_network"],
        "fixes": [fix.to_dict() for fix in fixes],
        "applied": count,
        "errors": errors,
    }


def is_correction_safe(original: str, corrected: str, min_similarity: float = 0.8) -> bool:
    original_words = _normalized_words(original)
    corrected_words = _normalized_words(corrected)
    if not corrected_words:
        return not original_words
    if original_words == corrected_words:
        return True
    ratio = difflib.SequenceMatcher(None, original_words, corrected_words).ratio()
    max_drift = max(2, int(len(original_words.split()) * 0.2))
    return (
        ratio >= min_similarity
        and abs(len(original_words.split()) - len(corrected_words.split())) <= max_drift
    )


def _normalized_words(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower().replace("ё", "е")))


def _batches(segments: list[Segment], maximum: int) -> list[list[int]]:
    budget = max(500, maximum)
    batches: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, segment in enumerate(segments):
        text = segment.text.strip()
        if not text:
            continue
        if current and size + len(text) > budget:
            batches.append(current)
            current, size = [], 0
        current.append(index)
        size += len(text)
    if current:
        batches.append(current)
    return batches


def _corrections(payload: object) -> dict[int, str]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("segments", [])
    else:
        items = []
    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        try:
            identifier = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        text = item["text"].strip()
        if text:
            result[identifier] = text
    return result


def build_article(
    transcript: Transcript, provider: LlamaProvider, settings: dict[str, Any]
) -> tuple[list[ArticleSection], dict[str, object]]:
    language = _prompt_language(transcript.language, str(settings["prompt_language"]))
    template = _prompt("article_default", language)
    chunks = _article_chunks(transcript.segments, int(settings["article_max_chars_chunk"]))
    sections: list[ArticleSection] = []
    reports: list[dict[str, object]] = []
    for chunk in chunks:
        source = " ".join(segment.text for segment in chunk)
        raw = ""
        accepted = False
        for attempt in range(2):
            try:
                raw = provider.complete(template.replace("{transcript}", source))
            except Exception as exc:
                reports.append({"ok": False, "reasons": [str(exc)]})
                break
            parsed = parse_markdown_sections(raw)
            body = " ".join(paragraph for item in parsed for paragraph in item["paragraphs"])
            report = faithfulness_report(source, body)
            if report["ok"] or attempt:
                reports.append(report)
                accepted = bool(report["ok"])
                for item in parsed:
                    sections.append(
                        ArticleSection(
                            item["title"], tuple(item["paragraphs"]), chunk[0].start, chunk[-1].end
                        )
                    )
                break
        if not accepted and not raw:
            continue
    report = {
        "schema_version": 1,
        "model": settings["model"],
        "chunks_total": len(chunks),
        "chunks_flagged": sum(not bool(item["ok"]) for item in reports),
        "sections": [section.to_dict() for section in sections],
        "faithfulness": reports,
    }
    return _merge_sections(sections), report


def faithfulness_report(source: str, edited: str) -> dict[str, object]:
    source_words = set(_normalized_words(source).split())
    edited_words = set(_normalized_words(edited).split())
    novel = len(edited_words - source_words) / len(edited_words) if edited_words else 0.0
    length = len(edited.strip()) / len(source.strip()) if source.strip() else 0.0
    coverage = len(source_words & edited_words) / len(source_words) if source_words else 1.0
    reasons: list[str] = []
    if novel > 0.15:
        reasons.append("new vocabulary exceeds 15%")
    if length > 1.15:
        reasons.append("text is longer than source")
    if length < 0.25 or coverage < 0.45:
        reasons.append("text is too abbreviated")
    return {
        "novel_word_ratio": round(novel, 4),
        "length_ratio": round(length, 4),
        "source_coverage": round(coverage, 4),
        "ok": not reasons,
        "reasons": reasons,
    }


class MarkdownSection(TypedDict):
    title: str
    paragraphs: list[str]


def parse_markdown_sections(text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    title = ""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.replace("```markdown", "").replace("```", "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            if paragraphs:
                sections.append({"title": title, "paragraphs": paragraphs})
            title, paragraphs = stripped[3:].strip(), []
        elif stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    if paragraphs:
        sections.append({"title": title, "paragraphs": paragraphs})
    return sections


def render_article_markdown(title: str, sections: list[ArticleSection]) -> str:
    lines = [f"# {title}", ""]
    for section in sections:
        if section.title:
            lines.extend((f"## {section.title}", ""))
        for paragraph in section.paragraphs:
            lines.extend((paragraph, ""))
    return "\n".join(lines).rstrip() + "\n"


def _article_chunks(segments: list[Segment], maximum: int) -> list[list[Segment]]:
    batches: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for segment in segments:
        if current and size + len(segment.text) > max(500, maximum):
            batches.append(current)
            current, size = [], 0
        current.append(segment)
        size += len(segment.text)
    if current:
        batches.append(current)
    return batches


def _merge_sections(sections: list[ArticleSection]) -> list[ArticleSection]:
    merged: list[ArticleSection] = []
    for section in sections:
        same_title = merged and merged[-1].title.casefold().strip(
            " ."
        ) == section.title.casefold().strip(" .")
        if same_title:
            previous = merged[-1]
            merged[-1] = ArticleSection(
                previous.title,
                previous.paragraphs + section.paragraphs,
                previous.start,
                section.end,
            )
        else:
            merged.append(section)
    return merged


def _prompt(name: str, language: str) -> str:
    root = files("studio.resources").joinpath("prompts")
    candidate = root.joinpath(language, f"{name}.txt")
    if not candidate.is_file():
        candidate = root.joinpath("ru", f"{name}.txt")
    return candidate.read_text(encoding="utf-8")


def _prompt_language(transcript_language: str, configured: str) -> str:
    if configured in {"ru", "en"}:
        return configured
    return "en" if transcript_language.lower().startswith("en") else "ru"


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
