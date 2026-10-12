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
from studio.llm.schemas import (
    ANY_OBJECT_SCHEMA,
    CLEANUP_DECISIONS_SCHEMA,
    EPISODE_CONTEXT_SCHEMA,
    FALLBACK_SCHEMAS,
    JUDGE_REVIEWS_SCHEMA,
    SCOUT_JSON_SCHEMA,
)
from studio.llm.session import model_identity
from studio.reels.analysis.audio_features import (
    analysis_audio,
    annotate_audio,
    audio_identity,
)
from studio.reels.analysis.candidate_extraction import normalize_candidate_list
from studio.reels.analysis.chunking import adaptive_overlap_seconds, build_analysis_chunks
from studio.reels.analysis.context import (
    build_transcript_digest,
    episode_metadata_path,
    format_chapters,
    format_episode_context,
    format_metadata_for_digest,
    stratified_batches,
)
from studio.reels.analysis.contracts import (
    AnalysisChunk,
    MomentRecord,
    coerce_moment_record,
    has_quote_evidence,
    replace_record,
)
from studio.reels.analysis.decisions import CLEANUP_EDITABLE, JUDGE_EDITABLE, apply_stage_decisions
from studio.reels.analysis.metadata import finalize_moment_list
from studio.reels.analysis.ranking import (
    assign_clip_types,
    dedupe_moments,
    overlap_ratio_of_shorter,
    rank_moments,
    ranking_value,
)
from studio.reels.analysis.refine import (
    build_cleanup_payload,
    cleanup_batches,
    limit_review_candidates,
)
from studio.reels.analysis.selection import (
    build_judge_payload,
    build_requirements_text,
    fit_scout_chunk_chars,
    guard_stage_output,
    percentile,
    quality_filters,
    rate,
    rejection_rows,
    requested_quotas,
    scale_quotas_to_duration,
    split_by_quality_filters,
    target_candidate_range,
    topic_diversity,
    unreachable_buckets,
)
from studio.reels.analysis.speaker_context import speaker_change_starts, speaker_context_identity
from studio.reels.analysis.validation import (
    annotate_speech_rate,
    apply_quote_verification,
    clamp_record_to_window,
    clamp_records_to_episode,
    enforce_quote_containment,
    quote_verification_settings,
    snap_records,
    split_by_quote_ratio,
)
from studio.reels.markdown import build_description_text, build_hashtags
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
        variant = str(reels["prompt_variant"]).strip().lower() or "default"
        prompts = [
            _prompt(language, name, variant)
            for language in ("ru", "en")
            for name in ("chunk", "cleanup", "judge", "context")
        ]
        return stable_fingerprint(
            "reels-select-v12",
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
        """Forge's staged selection: the LLM discovers, Python proves and selects.

        Scout candidates with verbatim quotes, verify the quotes, let cleanup
        keep/drop/merge by id and the judge score by id, then pass the final
        gates and select under duration-scaled quotas with MMR diversity.
        """
        reels = _settings(context.settings)
        path = _transcript_path(context.project)
        if path is None:
            raise ValueError("no transcript is available")
        transcript = Transcript.load(path)
        segments = transcript.to_dict()["segments"]
        index = TranscriptIndex.from_transcript(transcript)
        if not index:
            raise ValueError("reels selection requires timed transcript evidence")
        language = _language(transcript.language, str(reels["prompt_language"]))
        variant = str(reels["prompt_variant"]).strip().lower() or "default"
        duration = float(transcript.duration)
        clips_per_hour = float(reels["clips_per_hour"])
        quotas = scale_quotas_to_duration(
            requested_quotas(reels), duration_s=duration, clips_per_hour=clips_per_hour
        )
        requirements = build_requirements_text(reels, quotas)
        scout_requirements = build_requirements_text(reels, quotas, include_counts=False)
        target_total = sum(max(0, value) for value in quotas.values()) or int(
            reels["max_candidates"]
        )
        filters = quality_filters(reels)
        warnings = unreachable_buckets(quotas, filters)
        quote_conf = quote_verification_settings(
            {
                "min_ratio": reels["scout_quote_ratio"],
                "min_final_ratio": reels["min_quote_ratio"],
                "refine_boundaries": reels["quote_refine"],
            }
        )
        scout_prompt = _prompt(language, "chunk", variant)
        chunk_seconds = int(reels["chunk_seconds"])
        max_chars = int(reels["max_chars_chunk"])
        overlap_s = (
            int(reels["chunk_overlap_s"])
            if float(reels["chunk_overlap_s"]) >= 0
            else adaptive_overlap_seconds(chunk_seconds)
        )
        chunks = build_analysis_chunks(
            segments, chunk_seconds=chunk_seconds, max_chars=max_chars, overlap_seconds=overlap_s
        )
        llm = context.settings.get("llm", {})
        llm = llm if isinstance(llm, dict) else {}
        fitted = fit_scout_chunk_chars(
            [chunk.text for chunk in chunks],
            max_chars=max_chars,
            prompt=scout_prompt,
            ctx_size=int(llm.get("ctx_size") or 0),
            parallel=int(llm.get("parallel") or 1),
            n_predict=int(llm.get("n_predict") or 4096),
        )
        if fitted is not None:
            max_chars = fitted
            chunks = build_analysis_chunks(
                segments, chunk_seconds=chunk_seconds, max_chars=max_chars,
                overlap_seconds=overlap_s,
            )

        def schemas(role: str) -> tuple[dict[str, Any], ...]:
            """The role's answer schema, its simplified form and any object (Forge)."""
            if not reels["strict_json_schema"]:
                return ()
            full = {"context": EPISODE_CONTEXT_SCHEMA, "scout": SCOUT_JSON_SCHEMA,
                    "cleanup": CLEANUP_DECISIONS_SCHEMA, "judge": JUDGE_REVIEWS_SCHEMA}[role]
            return (full, FALLBACK_SCHEMAS[role], ANY_OBJECT_SCHEMA)

        def routed(role: str) -> LlamaProvider:
            return role_provider(
                context.settings,
                "reels",
                role,
                context.project.work_dir / "cache" / "llm",
                LlamaProvider,
            )

        failures: list[str] = []
        rejected: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        requests = JsonRequestBudget(
            int(reels["json_retries"]),
            int(reels["retry_budget"]),
            backoff_s=float(reels["retry_backoff_s"]),
        )
        episode_context = ""
        context_payload = None
        digest = ""
        metadata_context = ""
        metadata: Any = None
        speaker_starts: list[float] = []
        speaker_status = "disabled"
        metadata_path = episode_metadata_path(context.project)
        if metadata_path is not None:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata_context = format_metadata_for_digest(metadata)
            except (OSError, ValueError) as exc:
                metadata = None
                failures.append(f"episode_metadata: {exc}")
        if reels["episode_context"]:
            try:
                speaker_starts, speaker_status = speaker_change_starts(context.project, transcript)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                speaker_status = "failed"
                failures.append(f"speaker_context: {exc}")
            digest = build_transcript_digest(
                index, max_chars=int(reels["context_digest_chars"]),
                speaker_turns=speaker_starts,
            )
            try:

                def validate_context(payload: Any) -> Any:
                    format_episode_context(payload)
                    return payload

                context_payload = requests.request(
                    routed("context"),
                    _prompt(language, "context").replace(
                        "{transcript_digest}", (metadata_context + "\n\n" + digest).strip()
                    ),
                    validate_context,
                    schemas=schemas("context"),
                )
                episode_context = format_episode_context(context_payload)
            except Exception as exc:
                context_payload = None
                failures.append(f"episode_context: {exc}")
        chapters = format_chapters(metadata)
        if chapters:
            # The author's own structure: the scout sees where each topic
            # starts even when the overview call failed.
            episode_context = (episode_context + "\n" if episode_context else "") + chapters

        # Discovery: candidates with verbatim quotes; ids are assigned here.
        candidates: list[MomentRecord] = []
        scout_failures = 0
        for chunk in chunks:
            chunk_meta = {k: v for k, v in chunk.to_prompt_dict().items() if k != "text"}
            prompt = _render(
                scout_prompt,
                {
                    "requirements": scout_requirements,
                    "episode_context": episode_context,
                    "target_candidates": target_candidate_range(chunk.end - chunk.start),
                    "chunk_json": json.dumps(chunk_meta, ensure_ascii=False),
                    "transcript": chunk.text,
                },
            )
            try:
                found = normalize_candidate_list(
                    requests.request(routed("scout"), prompt, schemas=schemas("scout")),
                    stage="scout",
                )
            except Exception as exc:
                scout_failures += 1
                failures.append(f"{chunk.chunk_id}: {exc}")
                continue
            kept: list[MomentRecord] = []
            for record in found:
                if not has_quote_evidence(record):
                    continue
                bounded = clamp_record_to_window(
                    record, chunk.start, chunk.end, tolerance_s=float(reels["chunk_tolerance_s"])
                )
                if bounded is not None:
                    candidate_id = f"{chunk.chunk_id}_c{len(kept) + 1:02d}"
                    kept.append(_attach_chunk(bounded, chunk, candidate_id))
            candidates.extend(kept)
        if chunks and scout_failures == len(chunks):
            raise RuntimeError(
                f"scout failed on all {len(chunks)} chunk(s); the LLM server is likely down: "
                + "; ".join(failures[-3:])
            )
        counts["scouted"] = len(candidates)

        # Python proves: a quote that is not in the transcript costs no review.
        candidates = apply_quote_verification(candidates, index, **quote_conf)
        measured = [record for record in candidates if record.quote_match_ratio is not None]
        quote_stats = {
            "measured": len(measured),
            "exact": sum(1 for record in measured if record.quote_match_method == "exact"),
            "low_confidence": sum(
                1 for record in measured
                if (record.quote_match_ratio or 0.0) < quote_conf["min_final_ratio"]
            ),
        }
        candidates, unproven = split_by_quote_ratio(candidates, quote_conf["min_ratio"])
        rejected += rejection_rows(unproven, stage="scout", reason="quote_not_in_transcript")
        counts["after_quote_gate"] = len(candidates)

        # Chunks overlap by design: dedupe before capping the cleanup input.
        cleanup_size = max(1, int(reels["review_batch_size"]))
        cleanup_cap = max(cleanup_size, target_total * 2)
        if int(reels["cleanup_max_candidates"]) > 0:
            cleanup_cap = min(cleanup_cap, int(reels["cleanup_max_candidates"]))
        deduped = dedupe_moments(candidates)
        counts["after_dedupe"] = len(deduped)
        cleanup_input = sorted(deduped, key=ranking_value, reverse=True)[:cleanup_cap]
        counts["cleanup_input"] = len(cleanup_input)
        review_limits = [_limit_report("cleanup", deduped, cleanup_input, cleanup_cap)]
        review_reports: list[dict[str, Any]] = []
        guard = bool(reels["require_candidate_overlap"])
        if reels["cleanup"] and cleanup_input:
            ordered = sorted(
                cleanup_input, key=lambda record: (-ranking_value(record), record.start, record.end)
            )[: max(12, target_total * 3)]
            cleaned: list[MomentRecord] = []
            for offset, batch in enumerate(cleanup_batches(ordered, cleanup_size)):
                prompt = _render(
                    _prompt(language, "cleanup", variant),
                    {
                        "requirements": requirements,
                        "episode_context": episode_context,
                        "candidates_json": json.dumps(
                            build_cleanup_payload(batch), ensure_ascii=False
                        ),
                    },
                )
                cleaned.extend(self._review(
                    requests, routed("cleanup"), prompt, batch, "cleanup", offset,
                    review_reports, failures, schemas("cleanup"),
                ))
            cleaned = dedupe_moments(cleaned) if cleaned else list(ordered)
            cleaned, untraced = guard_stage_output(cleaned, cleanup_input, enabled=guard)
            counts["cleanup_untraceable"] = untraced
        else:
            cleaned = list(cleanup_input)
        counts["after_cleanup"] = len(cleaned)
        # A misbehaving cleanup can lose good candidates: top the pool up with
        # the best dropped ones that overlap nothing it kept.
        pool_floor = max(target_total + max(3, target_total // 3), len(cleaned))
        if len(cleaned) < pool_floor:
            pool = list(cleaned)
            for candidate in sorted(cleanup_input, key=ranking_value, reverse=True):
                if len(pool) >= pool_floor:
                    break
                if not any(overlap_ratio_of_shorter(candidate, kept) > 0 for kept in pool):
                    pool.append(candidate)
            cleaned = pool
        counts["after_top_up"] = len(cleaned)

        # Audio is measured on the short list, so the judge and the ranking
        # both get to use it.
        cleaned = [
            replace_record(record, audio_energy_db=None, audio_silence_ratio=None)
            for record in cleaned
        ]
        audio_report: dict[str, Any] = {"status": "disabled"}
        if reels["audio_features"]:
            try:
                with analysis_audio(context.project, transcript) as (
                    source_audio,
                    placement,
                    stream,
                ):
                    cleaned, audio_report = annotate_audio(
                        cleaned,
                        source_audio,
                        reels,
                        cache_dir=context.project.work_dir / "cache" / "audio_features",
                        placement=placement,
                        stream=stream,
                    )
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
                audio_report = {"status": "failed", "error": str(exc)}

        # Judge in stratified batches: every batch spans the quality range, so
        # its absolute scores stay comparable across calls.
        judged = list(cleaned)
        counts["judge_input"] = len(cleaned)
        if reels["judge"] and cleaned:
            pool = cleaned
            if int(reels["judge_max_candidates"]) > 0:
                pool, _over = limit_review_candidates(cleaned, int(reels["judge_max_candidates"]))
            review_limits.append(
                _limit_report("judge", cleaned, pool, int(reels["judge_max_candidates"])))
            judged = []
            for offset, batch in enumerate(
                stratified_batches(pool, max(1, int(reels["judge_batch_size"])))
            ):
                payload = (
                    build_judge_payload(
                        batch, index, max_candidates=len(batch),
                        head_seconds=float(reels["judge_head_s"]),
                        tail_seconds=float(reels["judge_tail_s"]),
                        max_excerpt_chars=int(reels["judge_excerpt_chars"]),
                    )
                    if reels["judge_context"]
                    else [record.to_dict() for record in batch]
                )
                prompt = _render(
                    _prompt(language, "judge", variant),
                    {
                        "requirements": requirements,
                        "episode_context": episode_context,
                        "candidates_json": json.dumps(payload, ensure_ascii=False),
                    },
                )
                judged.extend(self._review(
                    requests, routed("judge"), prompt, batch, "judge", offset,
                    review_reports, failures, schemas("judge"),
                ))
            if not judged:
                # The judge rejected everything: fall back to the pre-judge pool.
                judged = list(cleaned)
            judged, untraced = guard_stage_output(judged, cleaned, enabled=guard)
            counts["judge_untraceable"] = untraced
        counts["after_judge"] = len(judged)

        # Final gate: the stricter quote threshold, then boundary work that
        # keeps the located quote inside the clip.
        judged, weak = split_by_quote_ratio(judged, quote_conf["min_final_ratio"])
        rejected += rejection_rows(weak, stage="final", reason="quote_low_confidence")
        counts["after_final_gate"] = len(judged)
        snapped = snap_records(
            judged, index, enabled=bool(reels["snap_boundaries"]),
            max_shift_s=float(reels["snap_max_shift_s"]),
        )
        shifts = [
            abs(value)
            for before, after in zip(judged, snapped, strict=True)
            for value in (after.start - before.start, after.end - before.end)
        ]
        contained, lost = enforce_quote_containment(
            clamp_records_to_episode(snapped, duration), duration=duration
        )
        rejected += rejection_rows(lost, stage="final", reason="quote_outside_clip")
        annotated = annotate_speech_rate(contained, index)
        replay_report: dict[str, Any] = {"status": "disabled"}
        if reels["replay"]:
            from studio.reels.analysis.replay import annotate_replay

            annotated, replay_report = annotate_replay(
                annotated, context.project, transcript, metadata_path
            )
        typed, filtered_out = split_by_quality_filters(
            assign_clip_types(annotated, quotas), filters
        )
        for record, reason in filtered_out:
            rejected += rejection_rows([record], stage="selection", reason=reason)
        counts["after_quality_filters"] = len(typed)

        # Deterministic selection, exactly once.
        weights = reels["scoring_weights"]
        selected = rank_moments(
            typed,
            clip_type_quotas=quotas,
            scoring_weights=weights if isinstance(weights, dict) and weights else None,
            diversity_enabled=bool(reels["diversity"]),
            max_topic_similarity=float(reels["max_topic_similarity"]),
            # With duration-scaled quotas the total is the promise and the mix
            # a preference; fixed counts stay strict.
            fill_to_total=target_total if clips_per_hour > 0 else None,
            max_overlap_ratio=float(reels["max_overlap_ratio"]),
            mmr_lambda=float(reels["mmr_lambda"]),
        )
        if typed and not selected:
            warnings.append(
                f"no candidate matched the clip quotas {quotas} ({len(typed)} candidates)"
            )
        final, unproven_final = finalize_moment_list(selected)
        counts["final"] = len(final)

        directory = context.work_dir / "stages" / "reels"
        directory.mkdir(parents=True, exist_ok=True)
        moments = directory / "moments.json"
        summary = directory / "reels_summary.md"
        report = directory / "report.json"
        rejected_path = directory / "rejected_candidates.json"
        moments.write_text(
            json.dumps([record.to_dict() for record in final], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        summary.write_text(_summary(final), encoding="utf-8")
        rejected_path.write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        reasons: dict[str, int] = {}
        for row in rejected:
            reasons[str(row["rejection_reason"])] = reasons.get(str(row["rejection_reason"]), 0) + 1
        hours = duration / 3600.0 if duration else 0.0
        report.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "duration_s": duration,
                    "quotas": quotas,
                    "target_clips": target_total,
                    "chunks": len(chunks),
                    "chunk_overlap_s": overlap_s,
                    "chunk_max_chars": max_chars,
                    "prompt_language": language,
                    "prompt_variant": variant,
                    "warnings": warnings,
                    "failures": failures,
                    "counts": counts,
                    "candidate_survival_rate": {
                        "quote_gate": rate(counts["after_quote_gate"], counts["scouted"]),
                        "cleanup": rate(counts["after_cleanup"], counts["cleanup_input"]),
                        "judge": rate(counts["after_judge"], counts["judge_input"]),
                    },
                    "scout_candidates_per_hour": rate(counts["scouted"], hours),
                    "quote_exact_match_rate": rate(quote_stats["exact"], quote_stats["measured"]),
                    "quote_low_confidence_rate": rate(
                        quote_stats["low_confidence"], quote_stats["measured"]
                    ),
                    "boundary_shift_seconds": {
                        "mean": round(sum(shifts) / len(shifts), 3) if shifts else 0.0,
                        "p95": percentile(shifts, 0.95),
                    },
                    "duplicate_rate": rate(
                        counts["after_quote_gate"] - counts["after_dedupe"],
                        counts["after_quote_gate"],
                    ),
                    "topic_diversity": topic_diversity(final),
                    "quota_fill_rate": rate(len(final), target_total),
                    "rejections": reasons,
                    "unproven_final": len(unproven_final),
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
                    if context_payload is not None
                    else "failed"
                    if reels["episode_context"]
                    else "disabled",
                    "time_domain": transcript.time_domain.value,
                    "transcript_fingerprint": stable_fingerprint(transcript.to_dict()),
                    "quote_rejected": [
                        row for row in rejected
                        if row["rejection_reason"] in {
                            "quote_not_in_transcript", "quote_low_confidence",
                            "quote_outside_clip",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts = (moments, summary, report, rejected_path)
        outputs = {**context.project.outputs, "reels": list(artifacts)}
        return StageOutput(artifacts, {"outputs": outputs})

    @staticmethod
    def _review(
        requests: JsonRequestBudget,
        provider: LlamaProvider,
        prompt: str,
        batch: list[MomentRecord],
        stage: str,
        offset: int,
        reports: list[dict[str, Any]],
        failures: list[str],
        schemas: tuple[dict[str, Any], ...] = (),
    ) -> list[MomentRecord]:
        """One cleanup or judge request; a failed batch keeps its input."""
        try:
            outcome = apply_stage_decisions(
                batch,
                requests.request(provider, prompt, schemas=schemas),
                stage=stage,
                editable=CLEANUP_EDITABLE if stage == "cleanup" else JUDGE_EDITABLE,
                record_judge_score=stage == "judge",
            )
        except Exception as exc:
            failures.append(f"{stage}:{offset}: {exc}")
            reports.append({"stage": stage, "batch": offset, "mode": "failed",
                            "candidate_ids": [record.candidate_id for record in batch]})
            return list(batch)
        reports.append({
            "stage": stage,
            "batch": offset,
            "candidate_ids": [record.candidate_id for record in batch],
            "mode": outcome.mode,
            "kept": outcome.kept,
            "dropped": outcome.dropped,
            "merged": outcome.merged,
            "unmentioned": outcome.unmentioned,
            "untraceable": outcome.untraceable,
        })
        return outcome.records


def _limit_report(
    stage: str, offered: list[MomentRecord], reviewed: list[MomentRecord], limit: int
) -> dict[str, Any]:
    kept = {record.candidate_id for record in reviewed}
    return {
        "stage": stage,
        "input_count": len(offered),
        "review_count": len(reviewed),
        "limit": limit,
        "budget_excluded_ids": [record.candidate_id for record in offered
                                if record.candidate_id not in kept],
    }


def _render(template: str, values: dict[str, str]) -> str:
    """Fill every placeholder the template names; braces stay literal otherwise."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def _attach_chunk(record: MomentRecord, chunk: AnalysisChunk, candidate_id: str) -> MomentRecord:
    """Python owns identity; a speaker-less candidate takes the chunk's speaker."""
    payload = {
        **record.to_dict(),
        "source_chunk_ids": list(dict.fromkeys([*record.source_chunk_ids, chunk.chunk_id])),
        "candidate_id": candidate_id,
    }
    if chunk.speaker_set and not payload.get("speaker"):
        payload["speaker"] = chunk.speaker_set[0]
        payload["speaker_confidence"] = 0.5 if len(chunk.speaker_set) == 1 else 0.35
    return coerce_moment_record(payload) or record


def _settings(settings: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "base_url": "http://127.0.0.1:8080",
        "model": "local",
        "prompt_language": "auto",
        "chunk_seconds": 600,
        "max_chars_chunk": 6000,
        "prompt_variant": "default",
        "strict_json_schema": True,
        "max_candidates": 10,
        "target_min_s": 30.0,
        "target_max_s": 60.0,
        "clips_per_hour": 10.0,
        "story_count": 2,
        "story_max_s": 15.0,
        "reel_count": 3,
        "reel_max_s": 60.0,
        "long_reel_count": 1,
        "long_reel_max_s": 180.0,
        "highlight_count": 5,
        "min_score": 7.0,
        "min_duration_s": 15.0,
        "max_duration_s": 180.0,
        "chunk_overlap_s": -1.0,
        "chunk_tolerance_s": 3.0,
        "require_candidate_overlap": True,
        "scout_quote_ratio": 0.55,
        "quote_refine": True,
        "snap_boundaries": True,
        "snap_max_shift_s": 3.0,
        "context_digest_chars": 4000,
        "judge_context": True,
        "judge_batch_size": 14,
        "judge_head_s": 15.0,
        "judge_tail_s": 5.0,
        "judge_excerpt_chars": 260,
        "scoring_weights": {},
        "diversity": True,
        "max_topic_similarity": 0.5,
        "mmr_lambda": 0.7,
        "max_overlap_ratio": 0.2,
        "cleanup": True,
        "judge": True,
        "review_batch_size": 16,
        "cleanup_max_candidates": 0,
        "judge_max_candidates": 0,
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
        "qa": True,
        "preview_reel": True,
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


def _prompt(language: str, stage: str = "chunk", variant: str = "default") -> str:
    """The stage prompt in its variant (``chunk_a.txt``), else the default one."""
    folder = files("studio.resources").joinpath("prompts", language)
    for name in (f"{stage}_{variant}.txt", f"{stage}_default.txt"):
        candidate = folder.joinpath(name)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"no {stage} prompt for {language}")


def _language(transcript_language: str, configured: str) -> str:
    return (
        configured
        if configured in {"ru", "en"}
        else "en"
        if transcript_language.lower().startswith("en")
        else "ru"
    )


def _summary(records: list[MomentRecord]) -> str:
    """Forge's reels.md: time, score, why, hook, description and hashtags per moment."""
    lines = ["# Reels Suggestions", ""]
    for number, record in enumerate(records, 1):
        value = record.to_dict()
        description = build_description_text(value)
        hashtags = build_hashtags(value, description_text=description)
        lines.append(f"## {number}. {str(value.get('title', '')).strip()} "
                     f"[{value.get('clip_type', 'reel')}]")
        lines.append(f"Time: {_hms(record.start)}-{_hms(record.end)}")
        score = f"Score: {record.score:.1f}/10"
        if record.priority is not None:
            score += f" (priority {record.priority:.2f})"
        lines.append(score)
        if record.why.strip():
            lines.append(f"Why: {record.why.strip()}")
        if record.hook.strip():
            lines.append(f"Hook: {record.hook.strip()}")
        lines.append("")
        if description:
            lines.extend((description, ""))
        if hashtags:
            lines.extend((" ".join(hashtags), ""))
    return "\n".join(lines).rstrip() + "\n"


def _hms(seconds: float) -> str:
    hours, minutes, secs = int(seconds // 3600), int(seconds % 3600 // 60), int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
