"""Select timestamped reels candidates with the local LLM and Forge analysis contracts."""

from __future__ import annotations

import json
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

from studio.core.project import Project, stable_fingerprint
from studio.core.transcript import Transcript
from studio.core.transcript_index import TranscriptIndex
from studio.llm.providers import LlamaProvider
from studio.llm.retries import JsonRequestBudget
from studio.llm.roles import role_provider
from studio.llm.session import model_identity
from studio.reels.analysis.audio_features import (
    analysis_audio,
    annotate_audio,
    audio_identity,
)
from studio.reels.analysis.candidate_extraction import normalize_candidate_list
from studio.reels.analysis.chunking import build_analysis_chunks
from studio.reels.analysis.context import (
    build_transcript_digest,
    episode_metadata_path,
    format_episode_context,
    format_metadata_for_digest,
    stratified_batches,
)
from studio.reels.analysis.contracts import MomentRecord, replace_record
from studio.reels.analysis.decisions import CLEANUP_EDITABLE, JUDGE_EDITABLE, apply_stage_decisions
from studio.reels.analysis.ranking import dedupe_moments, rank_moments
from studio.reels.analysis.refine import (
    build_cleanup_payload,
    cleanup_batches,
    limit_review_candidates,
)
from studio.reels.analysis.speaker_context import speaker_change_starts, speaker_context_identity
from studio.reels.analysis.validation import (
    annotate_speech_rate,
    apply_quote_verification,
    clamp_record_to_window,
    clamp_records_to_episode,
    enforce_quote_containment,
    snap_records,
    split_by_quote_ratio,
)
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput


class ReelsStage:
    id = "reels"
    title = "Reels selection"
    after: tuple[str, ...] = ("timeline", "text", "program", "speakers")
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("reels", "llm.managed", "llm.model_path", "llm.roles")
    optional_after = ("text", "program", "speakers")
    gpu = GpuUse.LLM

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        reels = _settings(settings)
        if not reels["enabled"]:
            return []
        roles = settings.get("llm", {}).get("roles", {})
        enabled = ["scout"] + [role for role in ("cleanup", "judge") if reels[role]]
        if reels["episode_context"]:
            enabled.append("context")
        return [
            Requirement(
                "llama", str(roles.get(role, {}).get("model", reels["model"])), f"reels {role}"
            )
            for role in enabled
        ]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        reels = _settings(settings)
        if not reels["enabled"]:
            return Decision.skip("reels are disabled")
        if "proofread" in project.transcripts:
            return Decision.run({"transcript": "proofread"})
        if "edited" in project.transcripts:
            return Decision.run({"transcript": "edited"})
        if "timeline" in project.transcripts:
            return Decision.run({"transcript": "timeline"})
        return Decision.blocked("no transcript is available", "run timeline or text")

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        reels = _settings(settings)
        source = _transcript_path(project)
        content = source.read_text(encoding="utf-8") if source and source.is_file() else ""
        prompts = [
            _prompt(language, name)
            for language in ("ru", "en")
            for name in ("chunk", "cleanup", "judge", "context")
        ]
        return stable_fingerprint(
            "reels-select-v11",
            content,
            reels,
            prompts,
            model_identity(settings),
            _metadata_identity(project) if reels["episode_context"] or reels["replay"] else "",
            speaker_context_identity(project) if reels["episode_context"] else "",
            settings.get("llm", {}).get("roles", {}),
            audio_identity(project, Transcript.load(source))
            if reels["audio_features"] and source and source.is_file()
            else "",
        )

    def run(self, context: StageContext) -> StageOutput:
        reels = _settings(context.settings)
        path = _transcript_path(context.project)
        if path is None:
            raise ValueError("no transcript is available")
        transcript = Transcript.load(path)
        chunks = build_analysis_chunks(
            transcript.to_dict()["segments"],
            chunk_seconds=int(reels["chunk_seconds"]),
            max_chars=int(reels["max_chars_chunk"]),
        )
        template = _prompt(_language(transcript.language, str(reels["prompt_language"])))

        def routed(role: str) -> LlamaProvider:
            return role_provider(
                context.settings,
                "reels",
                role,
                context.project.work_dir / "cache" / "llm",
                LlamaProvider,
            )

        candidates: list[MomentRecord] = []
        index = TranscriptIndex.from_transcript(transcript)
        if not index:
            raise ValueError("reels selection requires timed transcript evidence")
        failures: list[str] = []
        requests = JsonRequestBudget(
            int(reels["json_retries"]),
            int(reels["retry_budget"]),
            backoff_s=float(reels["retry_backoff_s"]),
        )
        episode_context = ""
        context_payload = None
        digest = ""
        metadata_context = ""
        speaker_starts: list[float] = []
        speaker_status = "disabled"
        if reels["episode_context"]:
            try:
                speaker_starts, speaker_status = speaker_change_starts(context.project, transcript)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                speaker_status = "failed"
                failures.append(f"speaker_context: {exc}")
            digest = build_transcript_digest(index, speaker_turns=speaker_starts)
            metadata_path = episode_metadata_path(context.project)
            if metadata_path is not None:
                try:
                    metadata_context = format_metadata_for_digest(
                        json.loads(metadata_path.read_text(encoding="utf-8"))
                    )
                except (OSError, ValueError) as exc:
                    failures.append(f"episode_metadata: {exc}")
            try:

                def validate_context(payload: Any) -> Any:
                    format_episode_context(payload)
                    return payload

                context_payload = requests.request(
                    routed("context"),
                    _prompt(
                        _language(transcript.language, str(reels["prompt_language"])), "context"
                    ).replace("{transcript_digest}", (metadata_context + "\n\n" + digest).strip()),
                    validate_context,
                )
                episode_context = format_episode_context(context_payload)
            except Exception as exc:
                context_payload = None
                failures.append(f"episode_context: {exc}")
        for chunk in chunks:
            prompt = (
                template.replace("{transcript}", chunk.text)
                .replace("{r_min}", str(reels["target_min_s"]))
                .replace("{r_max}", str(reels["target_max_s"]))
                .replace("{count}", str(reels["max_candidates"]))
                .replace("{episode_context}", episode_context)
            )
            if episode_context and "{episode_context}" not in template:
                prompt = episode_context + "\n\n" + prompt
            try:
                found = normalize_candidate_list(
                    requests.request(routed("scout"), prompt), stage="scout"
                )
                for number, record in enumerate(found, 1):
                    bounded = clamp_record_to_window(record, chunk.start, chunk.end)
                    if bounded is not None:
                        candidates.append(
                            replace_record(bounded, candidate_id=f"{chunk.chunk_id}_c{number:03d}")
                        )
            except Exception as exc:
                failures.append(f"{chunk.chunk_id}: {exc}")
        candidates = apply_quote_verification(candidates, index)
        candidates, rejected = split_by_quote_ratio(candidates, float(reels["min_quote_ratio"]))
        review_reports: list[dict] = []
        review_limits: list[dict] = []
        for stage, editable in (("cleanup", CLEANUP_EDITABLE), ("judge", JUDGE_EDITABLE)):
            if not reels[stage]:
                continue
            before = len(candidates)
            candidates, excluded = limit_review_candidates(
                candidates, int(reels[f"{stage}_max_candidates"])
            )
            review_limits.append(
                {
                    "stage": stage,
                    "input_count": before,
                    "review_count": len(candidates),
                    "limit": int(reels[f"{stage}_max_candidates"]),
                    "budget_excluded_ids": [record.candidate_id for record in excluded],
                    "deduplicated_count": before - len(candidates) - len(excluded),
                }
            )
            reviewed: list[MomentRecord] = []
            batch_size = int(reels["review_batch_size"])
            batches = (
                stratified_batches(candidates, batch_size)
                if stage == "judge"
                else cleanup_batches(candidates, batch_size)
            )
            for offset, batch in enumerate(batches):
                payload = [
                    {
                        **record.to_dict(),
                        "excerpt_head": index.text_between(
                            record.start, min(record.end, record.start + 5)
                        ),
                        "excerpt_tail": index.text_between(
                            max(record.start, record.end - 5), record.end
                        ),
                    }
                    for record in batch
                ]
                if stage == "cleanup":
                    payload = build_cleanup_payload(batch)
                review_template = _prompt(
                    _language(transcript.language, str(reels["prompt_language"])), stage
                )
                prompt = (
                    review_template.replace(
                        "{candidates_json}", json.dumps(payload, ensure_ascii=False)
                    )
                    .replace("{requirements}", "Keep only grounded, complete moments.")
                    .replace("{episode_context}", episode_context)
                )
                if episode_context and "{episode_context}" not in review_template:
                    prompt = episode_context + "\n\n" + prompt
                try:
                    outcome = apply_stage_decisions(
                        batch,
                        requests.request(routed(stage), prompt),
                        stage=stage,
                        editable=editable,
                        record_judge_score=stage == "judge",
                    )
                    reviewed.extend(outcome.records)
                    review_reports.append(
                        {
                            "stage": stage,
                            "batch": offset,
                            "candidate_ids": [record.candidate_id for record in batch],
                            "mode": outcome.mode,
                            "kept": outcome.kept,
                            "dropped": outcome.dropped,
                            "merged": outcome.merged,
                            "unmentioned": outcome.unmentioned,
                            "untraceable": outcome.untraceable,
                        }
                    )
                except Exception as exc:
                    failures.append(f"{stage}:{offset}: {exc}")
                    reviewed.extend(batch)
                    review_reports.append(
                        {
                            "stage": stage,
                            "batch": offset,
                            "mode": "failed",
                            "candidate_ids": [record.candidate_id for record in batch],
                        }
                    )
            candidates = dedupe_moments(reviewed) if stage == "cleanup" else reviewed
        candidates = snap_records(candidates, index)
        candidates = clamp_records_to_episode(candidates, transcript.duration)
        candidates, lost_quotes = enforce_quote_containment(
            candidates, duration=transcript.duration
        )
        candidates = annotate_speech_rate(candidates, index)
        candidates = [
            replace_record(record, audio_energy_db=None, audio_silence_ratio=None)
            for record in candidates
        ]
        audio_report = {"status": "disabled"}
        if reels["audio_features"]:
            try:
                with analysis_audio(context.project, transcript) as (
                    source_audio,
                    placement,
                    stream,
                ):
                    candidates, audio_report = annotate_audio(
                        candidates,
                        source_audio,
                        reels,
                        cache_dir=context.project.work_dir / "cache" / "audio_features",
                        placement=placement,
                        stream=stream,
                    )
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                audio_report = {"status": "failed", "error": str(exc)}
        replay_report: dict[str, Any] = {"status": "disabled"}
        if reels["replay"]:
            from studio.reels.analysis.replay import annotate_replay

            candidates, replay_report = annotate_replay(
                candidates, context.project, transcript, episode_metadata_path(context.project)
            )
        ranked = rank_moments(
            candidates,
            clip_type_quotas={
                key: int(reels["max_candidates"])
                for key in ("story", "highlight", "reel", "long_reel")
            },
        )[: int(reels["max_candidates"])]
        directory = context.work_dir / "stages" / "reels"
        directory.mkdir(parents=True, exist_ok=True)
        moments = directory / "moments.json"
        summary = directory / "reels_summary.md"
        report = directory / "report.json"
        moments.write_text(
            json.dumps([record.to_dict() for record in ranked], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        summary.write_text(_summary(ranked), encoding="utf-8")
        report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "chunks": len(chunks),
                    "failures": failures,
                    "audio_features": audio_report,
                    "replay": replay_report,
                    "llm_requests": {
                        "attempts": requests.attempts,
                        "retries_used": requests.retries_used,
                        "retry_budget_remaining": requests.remaining,
                        "backoff_total_s": requests.backoff_total_s,
                    },
                    "reviews": review_reports,
                    "review_limits": review_limits,
                    "episode_context": context_payload,
                    "context_digest": digest,
                    "metadata_context": metadata_context,
                    "speaker_context": {"status": speaker_status, "change_starts": speaker_starts},
                    "context_status": "ok"
                    if episode_context
                    else "failed"
                    if reels["episode_context"]
                    else "disabled",
                    "time_domain": transcript.time_domain.value,
                    "transcript_fingerprint": stable_fingerprint(transcript.to_dict()),
                    "quote_rejected": [record.to_dict() for record in rejected + lost_quotes],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs = {**context.project.outputs, "reels": [moments, summary, report]}
        return StageOutput((moments, summary, report), {"outputs": outputs})


def _settings(settings: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "base_url": "http://127.0.0.1:8080",
        "model": "local",
        "prompt_language": "auto",
        "chunk_seconds": 600,
        "max_chars_chunk": 6000,
        "max_candidates": 10,
        "target_min_s": 30.0,
        "target_max_s": 60.0,
        "cleanup": True,
        "judge": True,
        "review_batch_size": 10,
        "cleanup_max_candidates": 100,
        "judge_max_candidates": 50,
        "json_retries": 1,
        "retry_budget": 5,
        "request_timeout_s": 600.0,
        "retry_backoff_s": 1.0,
        "audio_features": False,
        "audio_max_candidates": 50,
        "audio_noise_db": -30.0,
        "audio_silence_min_s": 0.35,
        "audio_timeout_s": 30.0,
        "episode_context": True,
        "min_quote_ratio": 0.75,
        "loudness_lufs": -14.0,
        "covers": True,
        "replay": True,
        "speaker_angles": True,
    }
    if isinstance(settings.get("reels"), dict):
        defaults.update(settings["reels"])
    return defaults


def _transcript_path(project: Project) -> Path | None:
    return next(
        (
            project.transcripts[key]
            for key in ("proofread", "edited", "timeline")
            if key in project.transcripts
        ),
        None,
    )


def _metadata_identity(project: Project) -> str:
    path = episode_metadata_path(project)
    if path is None:
        return ""
    try:
        return stable_fingerprint(str(path), path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stable_fingerprint(str(path), "unavailable")


def _prompt(language: str, stage: str = "chunk") -> str:
    return (
        files("studio.resources")
        .joinpath("prompts", language, f"{stage}_default.txt")
        .read_text(encoding="utf-8")
    )


def _language(transcript_language: str, configured: str) -> str:
    return (
        configured
        if configured in {"ru", "en"}
        else "en"
        if transcript_language.lower().startswith("en")
        else "ru"
    )


def _summary(records: list[MomentRecord]) -> str:
    lines = ["# Reels Suggestions", ""]
    for index, record in enumerate(records, 1):
        value = record.to_dict()
        lines.extend(
            (f"## {index}. {value['title']} [{value['clip_type']}]", "", value["quote"], "")
        )
    return "\n".join(lines)
