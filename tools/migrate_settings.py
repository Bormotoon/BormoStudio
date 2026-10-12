"""Translate WhisperSync JSON and Forge YAML settings into Studio settings.

The rules follow the settings mapping table (UNIFIED_APP_PLAN.md, section 6.3).
Only keys present in a source file are migrated, so legacy defaults never
override Studio defaults. Every source key ends up in the report: migrated,
or unmapped with its planned Studio section and the reason. The result is
validated with the same checks the application uses before it is written.

Usage::

    python -m tools.migrate_settings --whispersync ws.json --forge config.yaml \\
        --output _studio/settings.yaml --report migration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from studio.core.settings import SettingsError, _construct, validate_settings

Transform = Callable[[Any], Any]


def _same(value: Any) -> Any:
    return value


def _language(value: Any) -> Any:
    return None if value in (None, "", "auto") else value


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _gpu_layers(value: Any) -> int | None:
    # Forge passed no flag for 0, letting llama-server fit layers itself.
    return None if value in (None, "", 0) else int(value)


def _cache_ram(value: Any) -> int | str | None:
    text = str(value).strip().lower() if value is not None else ""
    if text in {"", "null", "none"}:
        return None
    return "auto" if text == "auto" else int(float(text))


def _download(value: Any) -> str:
    # Forge "auto" fetched video whenever cutting could use it; Studio exports
    # and reels need the picture, so auto keeps video.
    return "audio" if str(value or "").strip().lower() == "audio" else "video"


@dataclass(frozen=True, slots=True)
class Rule:
    source: str
    target: str | tuple[str, ...]
    transform: Transform = _same

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.target,) if isinstance(self.target, str) else self.target


# WhisperSync keys live at the top level of its JSON config.
WHISPERSYNC_RULES = (
    Rule("model", "transcribe.model"),
    Rule("device", "transcribe.device"),
    Rule("compute_type", "transcribe.compute_type"),
    Rule("language", "transcribe.language", _language),
    Rule("beam_size", "transcribe.beam_size"),
    Rule("batch_size", "transcribe.batch_size"),
    Rule("transcribe_mode", "transcribe.mode"),
    Rule("quality_beam_size", "transcribe.quality_beam_size"),
    Rule("initial_prompt", "transcribe.initial_prompt"),
    Rule("use_cache", "transcribe.use_cache"),
    Rule("default_strategy", "sync.strategy"),
    Rule("recorder_mode", "sync.recorder_mode"),
    Rule("boundary_flex", "sync.boundary_flex"),
    Rule("voice_segment_minutes", "sync.voice_segment_minutes", float),
    Rule("render_master_wav", "sync.master_wav"),
    Rule("crossfade_ms", "sync.master_crossfade_ms", float),
    Rule("acoustic_fallback_grid_s", "sync.acoustic_grid_s"),
    Rule("acoustic_fallback_window_s", "sync.acoustic_window_s"),
    Rule("acoustic_fallback_min_sharpness", "sync.acoustic_min_sharpness"),
    Rule("self_check_mode", "sync.self_check"),
    Rule("voice_enhance", "sync.voice_enhance"),
    Rule("ambience_track", "sync.ambience"),
    Rule("ambience_model", "sync.ambience_model"),
    Rule("detect_retakes", "roughcut.detect_retakes"),
    Rule("retake_min_words", "roughcut.retake_min_words"),
    Rule("retake_max_gap_s", "roughcut.retake_max_gap_s"),
    Rule("phrase_gap_threshold", "roughcut.phrase_gap_threshold"),
    Rule("fcpxml_version", "export.fcpxml_version"),
    Rule("probe_timeout_s", "scan.probe_timeout_s"),
    Rule("repetition_penalty", "transcribe.repetition_penalty", float),
    Rule("no_repeat_ngram_size", "transcribe.no_repeat_ngram_size", int),
)

# Planned section from table 6.3 for WhisperSync keys without a Studio field yet.
_WHISPERSYNC_PLANNED = {
    "transcribe": ("vad_filter", "best_of", "patience", "condition_on_previous_text",
                   "cache_dir", "cache_max_age_days"),
    "sync": ("audio_source_camera", "timebase_source", "camera_av_offset_ms",
             "camera_av_offset_ms_by_camera", "crossfade_enabled", "output_audio_format",
             "stretch_method", "seam_snap_max_s", "render_workers", "min_anchors",
             "anchor_min_confidence", "match_window_margin", "seed_max_occurrences",
             "seed_bin_width", "acoustic_max_lag_s", "gcc_eps", "acoustic_fallback",
             "alignment_max_k_deviation", "alignment_max_residual_ms",
             "alignment_min_coverage", "alignment_min_acoustic_points", "flex_window_s",
             "flex_min_sharpness", "flex_deadband_s", "flex_max_shift_s",
             "pause_duck_enabled", "pause_duck_db", "pause_duck_fade_ms",
             "pause_duck_min_pause_s", "self_check_transcribe_mode",
             "self_check_min_run_words", "self_check_shift_threshold_s",
             "self_check_min_content_words", "reuse_source_dir"),
    "roughcut.retakes": ("retake_similarity",),
    "export": ("save_transcripts", "output_dir"),
    "scan": ("video_exts", "audio_exts"),
}

FORGE_RULES = (
    Rule("transcription.model", "transcribe.model"),
    Rule("transcription.device", "transcribe.device"),
    Rule("transcription.compute_type", "transcribe.compute_type"),
    Rule("transcription.language", "transcribe.language", _language),
    Rule("transcription.beam_size", "transcribe.beam_size"),
    Rule("transcription.batch_size", "transcribe.batch_size"),
    Rule("transcription.mode", "transcribe.mode"),
    Rule("transcription.quality_beam_size", "transcribe.quality_beam_size"),
    Rule("transcription.initial_prompt", "transcribe.initial_prompt"),
    Rule("diarization.model", "speakers.model"),
    Rule("proofread.enabled", "text.proofread"),
    Rule("proofread.max_chars_chunk", "text.max_chars_chunk"),
    Rule("proofread.min_similarity", "text.min_similarity"),
    Rule("proofread.terms.enabled", "text.term_check"),
    Rule("article.enabled", "text.article"),
    Rule("article.max_chars_chunk", "text.article_max_chars_chunk"),
    Rule("prompts.language", ("text.prompt_language", "reels.prompt_language")),
    Rule("llama_cpp.chunk_seconds", "reels.chunk_seconds"),
    Rule("llama_cpp.max_chars_chunk", "reels.max_chars_chunk"),
    Rule("llama_cpp.timeout", "reels.request_timeout_s", float),
    Rule("llama_cpp.service.model_path", "llm.model_path"),
    Rule("llama_cpp.service.startup_timeout", "llm.startup_timeout_s", float),
    Rule("llama_cpp.service.n_gpu_layers", "llm.n_gpu_layers", _gpu_layers),
    Rule("llama_cpp.service.ctx_size", "llm.ctx_size", _optional_int),
    Rule("llama_cpp.service.batch_size", "llm.batch_size", _optional_int),
    Rule("llama_cpp.service.ubatch_size", "llm.ubatch_size", _optional_int),
    Rule("llama_cpp.service.threads", "llm.threads", _optional_int),
    Rule("llama_cpp.service.main_gpu", "llm.main_gpu", _optional_int),
    Rule("llama_cpp.service.parallel", "llm.parallel", _optional_int),
    Rule("llama_cpp.service.cache_ram_mb", "llm.cache_ram_mb", _cache_ram),
    Rule("llama_cpp.service.cache_ram_reserve_mb", "llm.cache_ram_reserve_mb", int),
    Rule("llama_cpp.service.cache_ram_min_mb", "llm.cache_ram_min_mb", int),
    Rule("llama_cpp.service.cache_ram_max_mb", "llm.cache_ram_max_mb", int),
    Rule("llama_cpp.service.ctx_checkpoints", "llm.ctx_checkpoints", _optional_int),
    Rule("llama_cpp.service.log_file", "llm.log_file", _text),
    Rule("llama_cpp.service.extra_args", "llm.extra_args",
         lambda value: [str(item) for item in value or []]),
    Rule("llama_cpp.temperature", "llm.temperature", float),
    Rule("llama_cpp.n_predict", "llm.n_predict", _optional_int),
    Rule("llama_cpp.watchdog.enabled", "llm.watchdog", bool),
    Rule("llama_cpp.watchdog.first_token_timeout", "llm.first_token_timeout_s", float),
    Rule("llama_cpp.watchdog.stall_timeout", "llm.stall_timeout_s", float),
    Rule("llama_cpp.watchdog.max_retries", "llm.max_retries", int),
    Rule("llama_cpp.role_overrides.scout.chunk_seconds", "reels.chunk_seconds", int),
    Rule("transcription.repetition_penalty", "transcribe.repetition_penalty", float),
    Rule("transcription.no_repeat_ngram_size", "transcribe.no_repeat_ngram_size", int),
    Rule("processing.analysis.json_retry", "reels.json_retries"),
    Rule("processing.analysis.json_retry_budget", "reels.retry_budget"),
    Rule("processing.analysis.quote_verification.min_final_ratio", "reels.min_quote_ratio"),
    Rule("processing.analysis.quote_verification.min_ratio", "reels.scout_quote_ratio", float),
    Rule("processing.analysis.quote_verification.refine_boundaries", "reels.quote_refine", bool),
    Rule("processing.analysis.cleanup_cap", "reels.review_batch_size", int),
    Rule("processing.analysis.chunk_overlap_s", "reels.chunk_overlap_s", float),
    Rule("processing.analysis.validation.chunk_tolerance_s", "reels.chunk_tolerance_s", float),
    Rule("processing.analysis.validation.require_candidate_overlap",
         "reels.require_candidate_overlap", bool),
    Rule("processing.analysis.boundary_snap.enabled", "reels.snap_boundaries", bool),
    Rule("processing.analysis.boundary_snap.max_shift_s", "reels.snap_max_shift_s", float),
    Rule("processing.analysis.judge_context.enabled", "reels.judge_context", bool),
    Rule("processing.analysis.judge_context.max_candidates", "reels.judge_batch_size", int),
    Rule("processing.analysis.judge_context.head_seconds", "reels.judge_head_s", float),
    Rule("processing.analysis.judge_context.tail_seconds", "reels.judge_tail_s", float),
    Rule("processing.analysis.judge_context.max_excerpt_chars", "reels.judge_excerpt_chars", int),
    Rule("processing.analysis.episode_context.max_digest_chars", "reels.context_digest_chars", int),
    Rule("processing.analysis.scoring.weights", "reels.scoring_weights",
         lambda value: {str(key): float(weight) for key, weight in (value or {}).items()}),
    Rule("processing.analysis.diversity.enabled", "reels.diversity", bool),
    Rule("processing.analysis.diversity.max_topic_similarity", "reels.max_topic_similarity", float),
    Rule("processing.analysis.diversity.mmr_lambda", "reels.mmr_lambda", float),
    Rule("processing.analysis.selection.max_overlap_ratio", "reels.max_overlap_ratio", float),
    Rule("processing.clips_per_hour", "reels.clips_per_hour", float),
    Rule("processing.clips.stories.count", "reels.story_count", int),
    Rule("processing.clips.stories.max_duration", "reels.story_max_s", float),
    Rule("processing.clips.reels.count", "reels.reel_count", int),
    Rule("processing.clips.reels.max_duration", "reels.reel_max_s", float),
    Rule("processing.clips.long_reels.count", "reels.long_reel_count", int),
    Rule("processing.clips.long_reels.max_duration", "reels.long_reel_max_s", float),
    Rule("processing.clips.highlights.moments_count", "reels.highlight_count", int),
    Rule("processing.quality_filters.min_score", "reels.min_score", float),
    Rule("processing.quality_filters.min_duration", "reels.min_duration_s", float),
    Rule("processing.quality_filters.max_duration", "reels.max_duration_s", float),
    Rule("prompts.variant", "reels.prompt_variant", _text),
    Rule("processing.clip_edges.lead_in_s", "reels.edge_lead_in_s", float),
    Rule("processing.clip_edges.tail_s", "reels.edge_tail_s", float),
    Rule("processing.clip_edges.guard_s", "reels.edge_guard_s", float),
    Rule("processing.clip_edges.max_extend_s", "reels.edge_max_extend_s", float),
    Rule("processing.clip_edges.max_trim_s", "reels.edge_max_trim_s", float),
    Rule("processing.quality_filters.render_rejected", "reels.render_rejected", bool),
    Rule("video.qa", "reels.qa", bool),
    Rule("video.qa_blackdetect", "reels.qa_blackdetect", bool),
    Rule("exports.webm", "reels.export_webm", bool),
    Rule("exports.gif", "reels.export_gif", bool),
    Rule("exports.audio_only", "reels.export_audio", bool),
    Rule("video.use_nvenc", "reels.encoder", lambda value: "auto" if value else "libx264"),
    Rule("video.video_bitrate", "reels.video_bitrate", _text),
    Rule("video.audio_bitrate", "reels.audio_bitrate", _text),
    Rule("video.preset", "reels.preset", _text),
    Rule("video.nvenc_cq", "reels.nvenc_cq", int),
    Rule("video.nvenc_preset", "reels.nvenc_preset", _text),
    Rule("subtitles.split_on_speaker", "reels.caption_split_on_speaker", bool),
    Rule("subtitles.speaker_colors", "reels.caption_speaker_colors",
         lambda value: [str(item) for item in value or []]),
    Rule("subtitles.vertical_align", "reels.caption_vertical_align", _text),
    Rule("subtitles.vertical_offset", "reels.caption_vertical_offset", float),
    Rule("subtitles.whisper_sync.enabled", "reels.caption_whisper_sync", bool),
    Rule("subtitles.whisper_sync.model", "reels.caption_sync_model", _text),
    Rule("subtitles.whisper_sync.min_match_ratio", "reels.caption_sync_min_match", float),
    Rule("subtitles.whisper_sync.apply_threshold_s", "reels.caption_sync_threshold_s", float),
    Rule("processing.analysis.episode_context.enabled", "reels.episode_context"),
    Rule("processing.analysis.audio_features.enabled", "reels.audio_features"),
    Rule("processing.analysis.audio_features.timeout_s", "reels.audio_timeout_s", float),
    Rule("processing.analysis.audio_features.silence_noise_db", "reels.audio_noise_db"),
    Rule("processing.analysis.audio_features.silence_min_s", "reels.audio_silence_min_s"),
    Rule("processing.analysis.audio_features.max_candidates", "reels.audio_max_candidates"),
    Rule("processing.reels_count", "reels.max_candidates"),
    Rule("processing.reel_min_duration", "reels.target_min_s", float),
    Rule("processing.reel_max_duration", "reels.target_max_s", float),
    Rule("subtitles.enabled", "reels.burn_subtitles"),
    Rule("subtitles.font_size_px", "reels.caption_font_size", int),
    Rule("subtitles.highlight", "reels.caption_highlight", _text),
    Rule("subtitles.text_case", "reels.caption_case", _text),
    Rule("subtitles.strip_punctuation", "reels.caption_punctuation", _text),
    Rule("subtitles.max_words_per_cue", "reels.caption_max_words", int),
    Rule("subtitles.max_lines", "reels.caption_max_lines", int),
    Rule("subtitles.max_chars_per_line", "reels.caption_max_chars", int),
    Rule("subtitles.max_width_ratio", "reels.caption_width_ratio", float),
    Rule("subtitles.highlight_color", "reels.caption_highlight_color", _text),
    Rule("subtitles.pause_split_s", "reels.caption_pause_split_s", float),
    Rule("subtitles.max_duration_s", "reels.caption_max_duration_s", float),
    Rule("subtitles.min_gap_s", "reels.caption_min_gap_s", float),
    Rule("subtitles.fade_min_gap_s", "reels.caption_fade_min_gap_s", float),
    Rule("subtitles.censor_words", "reels.caption_censor",
         lambda value: [str(item) for item in value or []]),
    Rule("subtitles.censor_style", "reels.caption_censor_style", _text),
    Rule("video.face_device", "reels.tracking_device"),
    Rule("video.active_speaker", "reels.active_speaker"),
    Rule("youtube.download", "fetch.download", _download),
    Rule("youtube.max_height", "fetch.max_height", int),
    Rule("youtube.cookies_file", "fetch.cookies_file", _text),
    Rule("youtube.retries", "fetch.retries", int),
    Rule("youtube.rate_limit", "fetch.rate_limit", _text),
    Rule("youtube.min_duration", "fetch.min_duration_s", float),
    Rule("youtube.max_duration", "fetch.max_duration_s", float),
    Rule("youtube.skip_live", "fetch.skip_live", bool),
    Rule("youtube.exclude", "fetch.exclude", lambda value: [str(item) for item in value or []]),
    Rule("autonomy.notify.when", "batch.notify_when"),
    Rule("autonomy.notify.webhook", "batch.notify_webhook", _text),
    Rule("autonomy.min_free_disk_gb", "batch.min_free_disk_gb", float),
)

# Forge sections by their planned Studio home (table 6.3) when not migrated.
_FORGE_PLANNED = {
    "transcription": "transcribe", "diarization": "speakers", "proofread": "text",
    "article": "text", "processing": "reels", "subtitles": "reels", "video": "reels",
    "audio": "reels", "exports": "reels", "prompts": "reels", "llama_cpp": "llm",
    "autonomy": "batch", "cli": "batch", "youtube": "fetch",
    "cache": "transcribe.cache + llm.cache", "host_memory": "contrib/hooks/host_memory + hooks",
    "paths": "project work folder + batch",
}

# Forge keys whose job Studio does differently; the report names the replacement.
_FORGE_REPLACED = {
    "youtube.sources": "pass each channel URL to `uvf channel`",
    "youtube.limit": "use `uvf channel --limit`",
    "youtube.download_dir": "each video gets its own project folder under the channel folder",
    "youtube.archive": "existing channel project folders are reused instead of an archive",
    "youtube.filename_template": "Studio names downloads source.<ext> per project",
    "youtube.api_key_env": "Studio lists channels through yt-dlp without an API key",
    "youtube.self_update": "reinstall the youtube module to update yt-dlp",
    "youtube.self_update_days": "reinstall the youtube module to update yt-dlp",
    "youtube.ydl_options": "arbitrary yt-dlp options are not passed through",
    "youtube.since": "date filters are not supported by flat channel listing",
    "youtube.until": "date filters are not supported by flat channel listing",
    "autonomy.lock_file": "Studio locks each project and channel folder itself",
    "autonomy.scheduling": "Studio runs stages in dependency order per project",
    "autonomy.log_dir": "logs and report.json live in each project work folder",
    "autonomy.log_keep_days": "logs live in each project work folder",
    "autonomy.runs_dir": "use `uvf batch --report` for a queue report",
    "autonomy.notify.command": "shell hooks are deferred (PROJECT_DECISIONS.md)",
    "llama_cpp.watchdog.log_interval": "Studio reports LLM progress as stage events",
    "llama_cpp.scout_parallelism": "Studio sends LLM requests one at a time (llm.parallel)",
    "llama_cpp.model_overrides": "use llm.roles and llm.role_options",
    "subtitles.ass_style": "save a look in the subtitle editor or choose reels.caption_preset",
    "subtitles.wrap_words": "reel captions always wrap into reels.caption_max_lines lines",
    "subtitles.keep_nosubs": "Studio always keeps the uncaptioned reel next to the captioned",
    "processing.analysis.llm_cache": "Studio always caches LLM answers in _studio/cache/llm",
    "processing.clips.highlights.count":
        "Forge never read it; reels.highlight_count sets the moments",
    "processing.reel_padding":
        "Studio needs word timings and fits edges to the speech (reels.edge_*)",
    "video.threads": "Studio renders reels one after another; each ffmpeg uses every core",
    "video.gpu_decode": "Studio decodes on the CPU; NVENC still encodes (reels.encoder)",
    "audio.listening_copy": "the review program (program.enabled) is the listening copy",
    "audio.delete_wav_after_analysis": "Studio keeps no analysis WAV next to the material",
    "processing.analysis.quote_verification.enabled":
        "Studio always verifies quotes; lower reels.scout_quote_ratio to keep more",
    "processing.analysis.audio_features.parallelism": "Studio measures candidates one by one",
    "processing.analysis.audio_features.cache":
        "Studio always caches measurements in _studio/cache/audio_features",
    "cli.quiet": "Studio prints JSON results; use --events for progress",
    "cli.verbose": "Studio prints JSON results; use --events for progress",
}

_FORGE_ROLES = {"scout": "scout", "cleanup_refine": "cleanup", "judge_metadata": "judge",
                "proofread": "proofread", "article": "article"}


@dataclass(slots=True)
class Migration:
    settings: dict[str, Any] = field(default_factory=dict)
    migrated: list[dict[str, Any]] = field(default_factory=list)
    unmapped: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def put(self, origin: str, source: str, target: str, value: Any) -> None:
        section, key = target.split(".", 1)
        bucket = self.settings.setdefault(section, {})
        if key in bucket and bucket[key] != value:
            self.conflicts.append({"target": target, "kept": value, "replaced": bucket[key],
                                   "source": f"{origin}:{source}"})
        bucket[key] = value
        self.migrated.append({"source": f"{origin}:{source}", "target": target, "value": value})

    def skip(self, origin: str, source: str, planned: str, reason: str) -> None:
        self.unmapped.append({"source": f"{origin}:{source}", "planned": planned,
                              "reason": reason})

    def report(self) -> dict[str, Any]:
        return {"schema_version": 1, "migrated": self.migrated, "unmapped": self.unmapped,
                "conflicts": self.conflicts}


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            result.update(_flatten(value, dotted + "."))
        else:
            result[dotted] = value
    return result


def migrate_whispersync(data: dict[str, Any], migration: Migration) -> None:
    rules = {rule.source: rule for rule in WHISPERSYNC_RULES}
    planned = {key: section for section, keys in _WHISPERSYNC_PLANNED.items() for key in keys}
    for key, value in data.items():
        rule = rules.get(key)
        if rule is None:
            migration.skip("whispersync", key, planned.get(key, "unknown"),
                           "no Studio setting yet" if key in planned else "unknown legacy key")
            continue
        if key == "crossfade_ms" and data.get("crossfade_enabled") is False:
            value = 0
        for target in rule.targets:
            migration.put("whispersync", key, target, rule.transform(value))


def migrate_forge(data: dict[str, Any], migration: Migration) -> None:
    rules = {rule.source: rule for rule in FORGE_RULES}
    flat = _flatten(data)
    for key, value in flat.items():
        section = key.split(".", 1)[0]
        rule = rules.get(key)
        if rule is not None:
            for target in rule.targets:
                migration.put("forge", key, target, rule.transform(value))
            continue
        if key == "llama_cpp.url" and isinstance(value, str):
            parts = urlsplit(value)
            base = f"{parts.scheme}://{parts.netloc}"
            for target in ("text.base_url", "reels.base_url", "roughcut.base_url"):
                migration.put("forge", key, target, base)
            if parts.port:
                migration.put("forge", key, "llm.port", parts.port)
            continue
        if key == "llama_cpp.service.auto_start":
            managed = bool(value) and bool(data.get("llama_cpp", {}).get("service", {}).get(
                "model_path"))
            migration.put("forge", key, "llm.managed", managed)
            continue
        if key == "subtitles.preset":
            # An empty Forge preset meant the shipped forge_subtitles.ass look.
            migration.put("forge", key, "reels.caption_preset", str(value or "forge"))
            continue
        if key == "subtitles.karaoke":
            if value and "highlight" not in data.get("subtitles", {}):
                migration.put("forge", key, "reels.caption_highlight", "karaoke")
            else:
                migration.skip("forge", key, "reels", "Studio uses reels.caption_highlight")
            continue
        if key in {"llama_cpp.service.cache_type_k", "llama_cpp.service.cache_type_v"}:
            migration.put("forge", key, f"llm.{key.rsplit('.', 1)[1]}", _text(value))
            if value:
                # Forge always started llama-server with flash attention on.
                migration.put("forge", key, "llm.flash_attn", "on")
            continue
        option = _role_option(key)
        if option is not None:
            option_role, name = option
            options = migration.settings.setdefault("llm", {}).setdefault("role_options", {})
            options.setdefault(option_role, {})[name] = float(value)
            migration.migrated.append({"source": f"forge:{key}",
                                       "target": f"llm.role_options.{option_role}.{name}",
                                       "value": float(value)})
            continue
        if key.startswith("llama_cpp.roles.") and isinstance(value, str):
            role = _FORGE_ROLES.get(key.rsplit(".", 1)[1])
            if role is not None:
                roles = migration.settings.setdefault("llm", {}).setdefault("roles", {})
                roles[role] = {"model": value}
                migration.migrated.append({"source": f"forge:{key}",
                                           "target": f"llm.roles.{role}.model", "value": value})
                continue
        if key == "diarization.enabled":
            if value:
                migration.put("forge", key, "speakers.method", "pyannote")
            else:
                migration.skip("forge", key, "speakers",
                               "Studio keeps speakers.method; microphones still attribute")
            continue
        if key == "video.vertical_crop":
            migration.put("forge", key, "reels.framing", "crop" if value else "source")
            continue
        if key == "video.smart_crop_face":
            migration.put("forge", key, "reels.tracking", bool(value))
            continue
        if key in _FORGE_REPLACED:
            migration.skip("forge", key, _FORGE_PLANNED[section], _FORGE_REPLACED[key])
            continue
        migration.skip("forge", key, _FORGE_PLANNED.get(section, "unknown"),
                       "no Studio setting yet" if section in _FORGE_PLANNED
                       else "unknown legacy section")


_ROLE_OPTION_NAMES = {"timeout": "timeout_s", "temperature": "temperature",
                      "n_predict": "n_predict"}


def _role_option(key: str) -> tuple[str, str] | None:
    """Forge per-role request options as (Studio role, option name)."""
    parts = key.split(".")
    if len(parts) == 4 and parts[:2] == ["llama_cpp", "role_overrides"]:
        role, name = _FORGE_ROLES.get(parts[2]), parts[3]
    elif len(parts) == 2 and parts[0] in {"proofread", "article"}:
        role, name = parts[0], parts[1]
    else:
        return None
    if role is None or name not in _ROLE_OPTION_NAMES:
        return None
    return role, _ROLE_OPTION_NAMES[name]


def _reconcile(migration: Migration) -> None:
    """Resolve dependencies Studio validates but the legacy apps did not."""
    reels = migration.settings.get("reels", {})
    if reels.get("tracking") and reels.get("framing") != "crop":
        reels["tracking"] = False
        migration.conflicts.append({"target": "reels.tracking", "kept": False,
                                    "reason": "tracking requires reels.framing=crop"})
    text = migration.settings.get("text", {})
    if text.get("term_check") and not text.get("proofread"):
        text["term_check"] = False
        migration.conflicts.append({"target": "text.term_check", "kept": False,
                                    "reason": "term_check requires proofread"})
    llm = migration.settings.get("llm", {})
    if llm.get("managed") and not llm.get("model_path"):
        llm["managed"] = False


def migrate(
    whispersync: dict[str, Any] | None = None, forge: dict[str, Any] | None = None
) -> Migration:
    """WhisperSync is applied first; Forge wins conflicts, which are reported."""
    migration = Migration()
    if whispersync:
        migrate_whispersync(whispersync, migration)
    if forge:
        migrate_forge(forge, migration)
    _reconcile(migration)
    try:
        validate_settings(_construct(migration.settings))
    except SettingsError as exc:
        raise SettingsError(f"migrated settings are invalid: {exc}") from exc
    return migration


def _load(path: Path, kind: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if kind == "whispersync" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise SettingsError(f"{path}: settings root must be a mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--whispersync", type=Path, help="WhisperSync JSON config")
    parser.add_argument("--forge", type=Path, help="Forge config.yaml")
    parser.add_argument("--output", type=Path, help="Studio settings.yaml to write")
    parser.add_argument("--report", type=Path, help="JSON report of every source key")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)
    if args.whispersync is None and args.forge is None:
        parser.error("give --whispersync and/or --forge")
    try:
        migration = migrate(
            _load(args.whispersync, "whispersync") if args.whispersync else None,
            _load(args.forge, "forge") if args.forge else None,
        )
    except (OSError, ValueError) as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 2
    document = yaml.safe_dump(migration.settings, allow_unicode=True, sort_keys=True)
    if args.output:
        if args.output.exists() and not args.force:
            print(f"{args.output} exists; use --force to replace it", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "# Migrated by tools/migrate_settings.py; see the report for unmapped keys.\n"
            + document, encoding="utf-8",
        )
    else:
        sys.stdout.write(document)
    report = migration.report()
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(f"migrated {len(report['migrated'])}, unmapped {len(report['unmapped'])}, "
          f"conflicts {len(report['conflicts'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
