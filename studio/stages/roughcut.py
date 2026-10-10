"""Deterministic rough-cut decisions from the timeline transcript."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from studio.core.audio_spans import Span, detect_silence, unknown_sound_spans, word_gaps
from studio.core.project import Project, stable_fingerprint
from studio.core.timeline import file_to_timeline
from studio.core.transcript import Transcript
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput
from studio.stages.retake_models import RetakeGroup, RetakeSettings
from studio.stages.retakes import detect_retakes
from studio.stages.roughcut_llm import LlmOptions, LlmResult, run_detectors

FILLERS = {"ну", "вот", "типа", "короче", "ээ", "эээ", "мм", "um", "uh"}


@dataclass(frozen=True, slots=True)
class Cut:
    start: float
    end: float
    reason: str
    confidence: float
    note: str
    accepted: bool = True

    @property
    def id(self) -> str:
        return stable_fingerprint("cut-v1", self.start, self.end, self.reason)[:16]


@dataclass(frozen=True, slots=True)
class Marker:
    at: float
    kind: str
    text: str


@dataclass(frozen=True, slots=True)
class EditList:
    mode: str
    keep: tuple[tuple[float, float], ...]
    cuts: tuple[Cut, ...]
    markers: tuple[Marker, ...]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "mode": self.mode,
            "keep": [
                {"start": start, "end": end, "camera_id": None}
                for start, end in self.keep
            ],
            "cuts": [{**asdict(cut), "id": cut.id} for cut in self.cuts],
            "markers": [asdict(marker) for marker in self.markers],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RoughcutStage:
    id = "roughcut"
    title = "Rough cut"
    after: tuple[str, ...] = ("timeline", "speakers")
    optional_after = ("speakers",)
    gpu = GpuUse.NONE

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        conf = _conf(settings)
        if not conf.get("llm", False):
            return []
        return [Requirement("llama", str(conf.get("model", "local")), "LLM rough-cut detectors")]

    def uses_llm(self, settings: dict[str, Any]) -> bool:
        return bool(_conf(settings).get("llm", False))

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        roughcut = settings.get("roughcut", {})
        enabled = roughcut.get("enabled", True) if isinstance(roughcut, dict) else True
        if not enabled:
            return Decision.skip("disabled by user")
        return Decision.run() if "timeline" in project.transcripts else Decision.blocked(
            "timeline transcript is missing", "run timeline"
        )

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        path = project.transcripts.get("timeline")
        content = path.read_text(encoding="utf-8") if path and path.is_file() else ""
        source = Transcript.load(path).source_audio if path and path.is_file() else None
        from studio.stages.transcribe import _sha256

        conf = _conf(settings)
        llm_identity: object = None
        if conf.get("llm", False):
            from studio.llm.session import model_identity
            from studio.stages.roughcut_llm import prompt_text

            llm = settings.get("llm", {})
            roles = llm.get("roles", {}) if isinstance(llm, dict) else {}
            llm_identity = (
                model_identity(settings),
                roles.get("roughcut", {}),
                [prompt_text(name, language) for name in ("roughcut_chunk", "roughcut_delivery")
                 for language in ("ru", "en")],
            )
        return stable_fingerprint(
            "roughcut-v5", content, project.placements,
            _sha256(source) if source and source.is_file() else "missing",
            settings.get("roughcut", {}), llm_identity,
        )

    def run(self, context: StageContext) -> StageOutput:
        transcript = Transcript.load(context.project.transcripts["timeline"])
        roughcut = context.settings.get("roughcut", {})
        conf = roughcut if isinstance(roughcut, dict) else {}
        source_id = transcript.metadata.get("source_asset_id")
        placement = next(
            (item for item in context.project.placements if item.asset_id == source_id), None
        )
        source = transcript.source_audio
        if not source.is_file():
            raise ValueError("roughcut energy source is missing")
        silence = detect_silence(
            source, threshold_db=float(conf.get("silence_threshold_db", -40.0)),
            duration_s=placement.duration_s if placement else transcript.duration,
        )
        if placement is not None:
            silence = [Span(file_to_timeline(span.start_s, placement),
                            file_to_timeline(span.end_s, placement)) for span in silence]
        retakes = RetakeSettings(
            detect_retakes=bool(conf.get("detect_retakes", True)),
            retake_min_words=int(conf.get("retake_min_words", 4)),
            retake_max_gap_s=float(conf.get("retake_max_gap_s", 6.0)),
            phrase_gap_threshold=float(conf.get("phrase_gap_threshold", 0.6)),
        )
        llm = None
        artifacts: list[Path] = []
        if conf.get("llm", False):
            llm = _run_llm(context, transcript, detect_retakes(transcript.words, retakes), conf)
            report = context.work_dir / "stages" / "roughcut" / "llm-report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(llm.report(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            artifacts.append(report)
        edit = build_edit_list(
            transcript,
            mode=str(conf.get("mode", "cut")),
            minimum_pause_s=float(conf.get("pause_min_s", 1.0)),
            keep_pause_s=float(conf.get("pause_keep_s", 0.4)),
            edge_pad_s=float(conf.get("head_tail_pad_s", 0.2)),
            silence=silence,
            retakes=retakes,
            llm=llm,
            llm_apply=str(conf.get("llm_apply", "suggest")),
            unknown_sound_min_s=float(conf.get("unknown_sound_min_s", 0.3)),
            audio_coverage=(Span(placement.offset_s,
                                 placement.offset_s + placement.duration_s * placement.k)
                            if placement else None),
        )
        overrides = context.project.work_dir / "edit-overrides.json"
        if overrides.is_file():
            decisions = json.loads(overrides.read_text(encoding="utf-8"))
            edit = apply_cut_overrides(edit, transcript.duration, decisions)
        output = context.work_dir / "stages" / "roughcut" / "edit.json"
        edit.save(output)
        outputs = {**context.project.outputs, "roughcut": [output]}
        return StageOutput((output, *artifacts), {"outputs": outputs})


def _conf(settings: dict[str, Any]) -> dict:
    roughcut = settings.get("roughcut", {})
    return roughcut if isinstance(roughcut, dict) else {}


def _run_llm(
    context: StageContext, transcript: Transcript, groups: list[RetakeGroup], conf: dict
) -> LlmResult:
    from studio.llm.providers import LlamaProvider
    from studio.llm.retries import JsonRequestBudget
    from studio.llm.roles import role_provider

    provider = role_provider(
        context.settings, "roughcut", "roughcut",
        context.project.work_dir / "cache" / "llm", LlamaProvider,
    )
    budget = JsonRequestBudget(
        int(conf.get("llm_json_retries", 1)), int(conf.get("llm_retry_budget", 5)),
    )
    language = str(conf.get("llm_prompt_language", "auto"))
    if language not in {"ru", "en"}:
        language = "en" if transcript.language.lower().startswith("en") else "ru"
    options = LlmOptions(
        detectors=tuple(conf.get("llm_detectors", ("start", "slips", "fillers", "delivery"))),
        language=language,
        min_confidence=float(conf.get("llm_min_confidence", 0.6)),
        max_cut_s=float(conf.get("llm_max_cut_s", 8.0)),
        max_start_s=float(conf.get("llm_max_start_s", 600.0)),
        chunk_chars=int(conf.get("llm_chunk_chars", 6000)),
    )
    return run_detectors(
        transcript, groups, lambda prompt: budget.request(provider, prompt), options
    )


def apply_cut_overrides(edit: EditList, duration: float, decisions: dict[str, bool]) -> EditList:
    from dataclasses import replace

    if not isinstance(decisions, dict) or not all(
        isinstance(key, str) and isinstance(value, bool) for key, value in decisions.items()
    ):
        raise ValueError("edit overrides must map cut IDs to booleans")
    cuts = tuple(replace(cut, accepted=decisions.get(cut.id, cut.accepted)) for cut in edit.cuts)
    return EditList(
        edit.mode, _complement(duration, list(cuts)) if edit.mode == "cut" else ((0.0, duration),),
        cuts, edit.markers,
    )


def review_cut(project: Project, cut_id: str, *, accepted: bool) -> None:
    """Persist a review decision and republish only the edit and its manifest."""
    import tempfile

    from studio.core.project import Artifact, StageManifest, describe_artifact
    from studio.core.publication import publish_files

    paths = project.outputs.get("roughcut", [])
    if not paths:
        raise ValueError("roughcut has no published decisions")
    path = paths[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    cuts = tuple(Cut(**{key: value for key, value in item.items() if key != "id"})
                 for item in data["cuts"])
    if cut_id not in {cut.id for cut in cuts}:
        raise ValueError(f"unknown cut ID: {cut_id}")
    transcript = Transcript.load(project.transcripts["timeline"])
    edit = EditList(data["mode"], (), cuts, tuple(Marker(**item) for item in data["markers"]))
    override_path = project.work_dir / "edit-overrides.json"
    decisions = (json.loads(override_path.read_text(encoding="utf-8"))
                 if override_path.is_file() else {})
    decisions[cut_id] = accepted
    edit = apply_cut_overrides(edit, transcript.duration, decisions)
    manifest_path = project.work_dir / "manifests" / "roughcut.json"
    manifest = StageManifest.load(manifest_path)
    with tempfile.TemporaryDirectory(prefix=".review-", dir=project.work_dir) as temporary:
        root = Path(temporary)
        updated = root / "edit.json"
        edit.save(updated)
        description = describe_artifact(root, updated)
        manifest.artifacts = [
            Artifact(artifact.path, description.sha256, description.size)
            if (project.work_dir / artifact.path).resolve() == path.resolve() else artifact
            for artifact in manifest.artifacts
        ]
        override = root / "overrides.json"
        override.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
        updated_manifest = root / "manifest.json"
        manifest.save(updated_manifest)
        publish_files(project.work_dir, [
            (updated, path), (override, override_path), (updated_manifest, manifest_path),
        ])


def build_edit_list(
    transcript: Transcript,
    *,
    mode: str,
    minimum_pause_s: float,
    keep_pause_s: float,
    edge_pad_s: float,
    silence: list[Span] | None = None,
    retakes: RetakeSettings | None = None,
    llm: LlmResult | None = None,
    llm_apply: str = "suggest",
    unknown_sound_min_s: float = 0.3,
    audio_coverage: Span | None = None,
) -> EditList:
    if mode not in {"cut", "markers"}:
        raise ValueError("roughcut mode must be cut or markers")
    words = transcript.words
    if not words:
        sounds = unknown_sound_spans(
            [], silence, duration_s=transcript.duration,
            minimum_s=unknown_sound_min_s, coverage=audio_coverage,
        ) if silence is not None else []
        return EditList(mode, ((0.0, transcript.duration),), (), tuple(
            Marker(span.start_s, "unknown-sound", f"untranscribed audio {span.duration_s:.2f}s")
            for span in sounds
        ))
    cuts: list[Cut] = []
    def quiet(start: float, end: float) -> bool:
        return silence is None or any(
            span.start_s <= start and span.end_s >= end for span in silence
        )

    first = min(word.start for word in words)
    last = max(word.end for word in words)
    if first > edge_pad_s and quiet(0.0, first - edge_pad_s):
        cuts.append(Cut(0.0, first - edge_pad_s, "head", 1.0, "before first word"))
    if transcript.duration > last + edge_pad_s and quiet(last + edge_pad_s, transcript.duration):
        cuts.append(
            Cut(last + edge_pad_s, transcript.duration, "tail", 1.0, "after last word")
        )
    for gap in word_gaps(words, minimum_s=minimum_pause_s):
        trim = max(0.0, gap.duration_s - keep_pause_s)
        start = gap.start_s + keep_pause_s / 2
        if not quiet(start, start + trim):
            continue
        cuts.append(
            Cut(start, start + trim, "pause", 0.95, f"pause {gap.duration_s:.2f}s")
        )
    def inside_word(at: float) -> bool:
        return any(word.start < at < word.end for word in words)

    llm_markers: list[Marker] = []
    delivery = llm.delivery if llm is not None else {}
    for index, group in enumerate(detect_retakes(words, retakes or RetakeSettings())):
        count = len(group.takes)
        choice = delivery.get(index)
        if choice is not None and choice[0] != count - 1 and llm_apply == "cut":
            take, reason, confidence = choice
            chosen = group.takes[take]
            spans = [(group.takes[0].start, chosen.start)]
            if take + 1 < count:
                spans.append((group.takes[take + 1].start, group.takes[-1].end))
            note = f"delivery: keep take {take + 1} of {count}: {chosen.text} — {reason}"
            for start, end in spans:
                if end > start and not inside_word(start) and not inside_word(end):
                    cuts.append(Cut(start, end, "llm", confidence, note))
            continue
        if choice is not None and choice[0] != count - 1:
            llm_markers.append(Marker(
                group.takes[choice[0]].start, "llm-delivery",
                f"LLM prefers take {choice[0] + 1} of {count}: {choice[1]}",
            ))
        start, end = group.takes[0].start, group.keeper.start
        # Malformed/overlapping word timestamps must not turn a restart into
        # a cut through unrelated speech.
        if end > start and not inside_word(end):
            cuts.append(Cut(
                start, end, "retake", 0.8,
                f"keep take {count} of {count}: {group.keeper.text}",
            ))
    deterministic = list(cuts)
    for finding in llm.findings if llm is not None else []:
        if finding.detector == "fillers":
            llm_markers.append(Marker(
                finding.start, "llm-filler", f"{finding.quote} — {finding.reason}"
            ))
            continue
        if any(cut.start <= finding.start and finding.end <= cut.end for cut in deterministic):
            continue
        cuts.append(Cut(
            finding.start, finding.end, "llm", finding.confidence,
            f"{finding.detector}: {finding.quote} — {finding.reason}",
            accepted=llm_apply == "cut",
        ))
    # Every detector shares the same boundary guard, including retakes and LLM findings.
    cuts = [cut for cut in cuts if 0 <= cut.start < cut.end <= transcript.duration
            and not inside_word(cut.start) and not inside_word(cut.end)]
    cuts.sort(key=lambda cut: cut.start)
    fillers = [
        *(Marker(word.start, "filler", word.text)
          for word in words
          if word.text.casefold().strip(".,!?…") in FILLERS),
        *(Marker(left.start, "filler", f"{left.text} {right.text}")
          for left, right in zip(words, words[1:], strict=False)
          if left.text.casefold().strip(".,!?…") == "как"
          and right.text.casefold().strip(".,!?…") == "бы"
          and 0 <= right.start - left.end <= 0.3),
    ]
    # An LLM filler marker at the same word replaces the heuristic one.
    llm_filler_starts = {marker.at for marker in llm_markers if marker.kind == "llm-filler"}
    markers = tuple(sorted((
        *(marker for marker in fillers if marker.at not in llm_filler_starts),
        *llm_markers,
        *(Marker(span.start_s, "unknown-sound",
                 f"untranscribed audio {span.duration_s:.2f}s")
          for span in (unknown_sound_spans(
              words, silence, duration_s=transcript.duration,
              minimum_s=unknown_sound_min_s, coverage=audio_coverage,
          ) if silence is not None else [])),
    ), key=lambda marker: marker.at))
    keep = (
        _complement(transcript.duration, cuts)
        if mode == "cut"
        else ((0.0, transcript.duration),)
    )
    return EditList(mode, keep, tuple(cuts), markers)


def _complement(duration: float, cuts: list[Cut]) -> tuple[tuple[float, float], ...]:
    keep: list[tuple[float, float]] = []
    cursor = 0.0
    for cut in cuts:
        if cut.accepted and cut.start > cursor:
            keep.append((cursor, cut.start))
        if cut.accepted:
            cursor = max(cursor, cut.end)
    if cursor < duration:
        keep.append((cursor, duration))
    return tuple((start, end) for start, end in keep if end > start)
