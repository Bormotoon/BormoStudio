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
)

# Planned section from table 6.3 for WhisperSync keys without a Studio field yet.
_WHISPERSYNC_PLANNED = {
    "transcribe": ("vad_filter", "best_of", "patience", "condition_on_previous_text",
                   "repetition_penalty", "no_repeat_ngram_size", "cache_dir",
                   "cache_max_age_days"),
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
    Rule("processing.analysis.json_retry", "reels.json_retries"),
    Rule("processing.analysis.json_retry_budget", "reels.retry_budget"),
    Rule("processing.analysis.quote_verification.min_final_ratio", "reels.min_quote_ratio"),
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
    Rule("video.face_device", "reels.tracking_device"),
    Rule("video.active_speaker", "reels.active_speaker"),
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
        migration.skip("forge", key, _FORGE_PLANNED.get(section, "unknown"),
                       "no Studio setting yet" if section in _FORGE_PLANNED
                       else "unknown legacy section")


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
