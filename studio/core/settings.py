"""Typed, layered Studio settings with fail-fast validation."""

from __future__ import annotations

import math
import types
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class SettingsError(ValueError):
    pass


@dataclass(slots=True)
class ScanSettings:
    grouping: str = "auto"
    probe_timeout_s: float = 30.0


@dataclass(slots=True)
class SyncSettings:
    mode: str = "auto"
    strategy: int = 3
    max_drift_ms: float = 20.0
    acoustic_grid_s: float = 30.0
    acoustic_window_s: float = 8.0
    acoustic_min_sharpness: float = 50.0
    recorder_mode: str = "best"
    session_offsets: dict[str, float] = field(default_factory=dict)
    reference_camera: str = "auto"
    boundary_flex: bool = False
    verify: bool = False
    voice_segment_minutes: float = 0.0
    master_wav: bool = False
    master_crossfade_ms: float = 10.0
    self_check: str = "off"
    voice_enhance: str = "off"
    ambience: bool = False
    ambience_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


@dataclass(slots=True)
class TranscribeSettings:
    model: str = "large-v3"
    language: str | None = "ru"
    device: str = "auto"
    compute_type: str = "auto"
    mode: str = "fast"
    batch_size: int = 16
    beam_size: int = 5
    quality_beam_size: int = 10
    initial_prompt: str = ""
    use_cache: bool = True
    glossary: list[str] = field(default_factory=list)
    hotwords: str = ""
    audio_streams: dict[str, int] = field(default_factory=dict)
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    keep_fillers: bool = True


@dataclass(slots=True)
class RoughcutSettings:
    enabled: bool = True
    mode: str = "cut"
    pause_min_s: float = 1.0
    pause_keep_s: float = 0.4
    silence_threshold_db: float = -40.0
    unknown_sound_min_s: float = 0.3
    head_tail_pad_s: float = 0.2
    detect_retakes: bool = True
    retake_min_words: int = 4
    retake_max_gap_s: float = 6.0
    phrase_gap_threshold: float = 0.6
    llm: bool = False
    llm_detectors: list[str] = field(
        default_factory=lambda: ["start", "slips", "fillers", "delivery"]
    )
    llm_apply: str = "suggest"
    llm_min_confidence: float = 0.6
    llm_max_cut_s: float = 8.0
    llm_max_start_s: float = 600.0
    llm_chunk_chars: int = 6000
    llm_prompt_language: str = "auto"
    llm_json_retries: int = 1
    llm_retry_budget: int = 5
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    request_timeout_s: float = 600.0


@dataclass(slots=True)
class ExportSettings:
    targets: list[str] = field(default_factory=lambda: ["fcpxml", "xmeml"])
    fcpxml_version: str = "1.9"


@dataclass(slots=True)
class ProgramSettings:
    enabled: bool = False
    burn_subtitles: bool = False
    encoder: str = "auto"
    fps: str = "auto"
    speaker_cameras: dict[str, str] = field(default_factory=dict)
    min_shot_s: float = 1.0
    cut_lead_s: float = 0.2
    max_switches: int = 3
    switch_window_s: float = 6.0
    max_shot_s: float = 0.0
    wide_insert_s: float = 3.0
    crossfade_ms: float = 10.0


@dataclass(slots=True)
class ComputeSettings:
    allow_cpu: bool = False


@dataclass(slots=True)
class SpeakerSettings:
    method: str = "auto"
    margin_db: float = 6.0
    silence_floor_db: float = -60.0
    step_s: float = 0.05
    min_turn_s: float = 0.3
    max_gap_s: float = 0.3
    tracks: dict[str, str] = field(default_factory=dict)
    model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cpu"


@dataclass(slots=True)
class TextSettings:
    proofread: bool = False
    article: bool = False
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    prompt_language: str = "auto"
    max_chars_chunk: int = 4000
    article_max_chars_chunk: int = 6000
    min_similarity: float = 0.8
    term_check: bool = False
    term_check_network: bool = False
    term_fixes: dict[str, str] = field(default_factory=dict)
    term_max_candidates: int = 10


@dataclass(slots=True)
class ReelsSettings:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    prompt_language: str = "auto"
    chunk_seconds: int = 600
    max_chars_chunk: int = 6000
    max_candidates: int = 10
    target_min_s: float = 30.0
    target_max_s: float = 60.0
    cleanup: bool = True
    judge: bool = True
    review_batch_size: int = 10
    cleanup_max_candidates: int = 100
    judge_max_candidates: int = 50
    json_retries: int = 1
    retry_budget: int = 5
    request_timeout_s: float = 600.0
    retry_backoff_s: float = 1.0
    audio_features: bool = False
    audio_max_candidates: int = 50
    audio_noise_db: float = -30.0
    audio_silence_min_s: float = 0.35
    audio_timeout_s: float = 30.0
    episode_context: bool = True
    min_quote_ratio: float = 0.75
    render: bool = False
    render_fps: str = "25"
    burn_subtitles: bool = False
    framing: str = "source"
    width: int = 1080
    height: int = 1920
    crop_x: float = 0.5
    tracking: bool = False
    tracking_device: str = "cuda"
    active_speaker: bool = True
    # Integrated loudness target for rendered reels (0 keeps source loudness).
    loudness_lufs: float = -14.0
    covers: bool = True
    # Rank with the YouTube "most replayed" heatmap of fetched episodes.
    replay: bool = True


@dataclass(slots=True)
class LlmSettings:
    managed: bool = False
    executable: str = "llama-server"
    model_path: str = ""
    port: int = 8080
    startup_timeout_s: float = 120.0
    roles: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class FetchSettings:
    url: str = ""
    download: str = "video"  # "video" or "audio"
    max_height: int = 0  # 0 = best available
    cookies_file: str = ""
    retries: int = 3
    rate_limit: str = ""  # yt-dlp --limit-rate, e.g. "5M"
    # Channel discovery filters; 0 disables a duration bound.
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0
    skip_live: bool = True
    exclude: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BatchSettings:
    notify_when: str = "failure"
    notify_webhook: str = ""
    notify_desktop: bool = False
    # Stop starting projects when the source volume has less free space (0 = off).
    min_free_disk_gb: float = 0.0


@dataclass(slots=True)
class Settings:
    llm: LlmSettings = field(default_factory=LlmSettings)
    scan: ScanSettings = field(default_factory=ScanSettings)
    sync: SyncSettings = field(default_factory=SyncSettings)
    transcribe: TranscribeSettings = field(default_factory=TranscribeSettings)
    roughcut: RoughcutSettings = field(default_factory=RoughcutSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    program: ProgramSettings = field(default_factory=ProgramSettings)
    compute: ComputeSettings = field(default_factory=ComputeSettings)
    speakers: SpeakerSettings = field(default_factory=SpeakerSettings)
    text: TextSettings = field(default_factory=TextSettings)
    reels: ReelsSettings = field(default_factory=ReelsSettings)
    fetch: FetchSettings = field(default_factory=FetchSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_settings(paths: list[Path], overrides: list[str] | None = None) -> Settings:
    merged: dict[str, Any] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise SettingsError(f"{path}: invalid YAML: {exc}") from exc
        if not isinstance(value, dict):
            raise SettingsError(f"{path}: settings root must be a mapping")
        merged = _merge(merged, value)
    for override in overrides or []:
        _set_override(merged, override)
    settings = _construct(merged)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    for section_name in Settings.__dataclass_fields__:
        section = getattr(settings, section_name)
        for name, expected in get_type_hints(type(section)).items():
            if not _matches_type(getattr(section, name), expected):
                raise SettingsError(f"{section_name}.{name} has an invalid value type")
    if not 1 <= settings.llm.port <= 65535:
        raise SettingsError("llm.port must be between 1 and 65535")
    _positive("llm.startup_timeout_s", settings.llm.startup_timeout_s)
    if settings.llm.managed and (
        not settings.llm.model_path.strip() or not settings.llm.executable.strip()
    ):
        raise SettingsError("managed llama requires executable and model_path")
    _one_of("scan.grouping", settings.scan.grouping, {"auto", "folders", "names"})
    _positive("scan.probe_timeout_s", settings.scan.probe_timeout_s)
    _one_of("sync.mode", settings.sync.mode, {"auto", "camera", "simple", "complex"})
    if settings.sync.strategy not in {1, 2, 3}:
        raise SettingsError("sync.strategy must be 1, 2, or 3")
    _non_negative("sync.max_drift_ms", settings.sync.max_drift_ms)
    _non_negative("sync.voice_segment_minutes", settings.sync.voice_segment_minutes)
    _non_negative("sync.master_crossfade_ms", settings.sync.master_crossfade_ms)
    _one_of("sync.recorder_mode", settings.sync.recorder_mode, {"best", "all"})
    if any(not key.strip() or not math.isfinite(value)
           for key, value in settings.sync.session_offsets.items()):
        raise SettingsError("sync.session_offsets requires names and finite seconds")
    if not settings.sync.reference_camera.strip():
        raise SettingsError("sync.reference_camera must not be empty")
    _one_of("sync.self_check", settings.sync.self_check, {"off", "warn", "repair"})
    if settings.sync.self_check == "repair" and settings.sync.voice_enhance != "off":
        raise SettingsError("sync.self_check=repair currently requires voice_enhance=off")
    _one_of(
        "sync.voice_enhance",
        settings.sync.voice_enhance,
        {"off", "denoise", "denoise_dereverb", "resemble"},
    )
    for name in ("acoustic_grid_s", "acoustic_window_s", "acoustic_min_sharpness"):
        _positive(f"sync.{name}", getattr(settings.sync, name))
    _one_of("transcribe.device", settings.transcribe.device, {"auto", "cuda", "cpu"})
    _one_of("transcribe.mode", settings.transcribe.mode, {"fast", "quality"})
    _positive("transcribe.repetition_penalty", settings.transcribe.repetition_penalty)
    if any(not key.strip() or value < 0
           for key, value in settings.transcribe.audio_streams.items()):
        raise SettingsError("transcribe.audio_streams requires names and non-negative indices")
    if settings.transcribe.no_repeat_ngram_size < 0:
        raise SettingsError("transcribe.no_repeat_ngram_size must be non-negative")
    if settings.transcribe.batch_size < 1:
        raise SettingsError("transcribe.batch_size must be at least 1")
    for name in ("batch_size", "beam_size", "quality_beam_size"):
        value = getattr(settings.transcribe, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SettingsError(f"transcribe.{name} must be a positive integer")
    if not isinstance(settings.transcribe.glossary, list) or not all(
        isinstance(item, str) for item in settings.transcribe.glossary
    ):
        raise SettingsError("transcribe.glossary must be a list of strings")
    _one_of("roughcut.mode", settings.roughcut.mode, {"cut", "markers"})
    _validate_roughcut_llm(settings.roughcut)
    _one_of("speakers.method", settings.speakers.method, {"auto", "mics", "pyannote", "off"})
    _one_of("speakers.device", settings.speakers.device, {"cpu", "cuda"})
    _non_negative("speakers.margin_db", settings.speakers.margin_db)
    _positive("speakers.step_s", settings.speakers.step_s)
    _non_negative("speakers.min_turn_s", settings.speakers.min_turn_s)
    _non_negative("speakers.max_gap_s", settings.speakers.max_gap_s)
    if settings.speakers.step_s > 1:
        raise SettingsError("speakers.step_s must not exceed 1 second")
    if not math.isfinite(settings.speakers.silence_floor_db) or not (
        -100 <= settings.speakers.silence_floor_db <= 0
    ):
        raise SettingsError("speakers.silence_floor_db must be between -100 and 0")
    _one_of("program.encoder", settings.program.encoder, {"auto", "nvenc", "libx264", "cpu"})
    _non_negative("program.min_shot_s", settings.program.min_shot_s)
    _non_negative("program.crossfade_ms", settings.program.crossfade_ms)
    lufs = settings.reels.loudness_lufs
    if not isinstance(lufs, (int, float)) or isinstance(lufs, bool) or not (
        lufs == 0 or -70 <= lufs <= -5
    ):
        raise SettingsError("reels.loudness_lufs must be 0 (off) or between -70 and -5")
    if settings.fetch.download not in {"video", "audio"}:
        raise SettingsError("fetch.download must be video or audio")
    for name in ("max_height", "retries", "min_duration_s", "max_duration_s"):
        _non_negative(f"fetch.{name}", getattr(settings.fetch, name))
    if 0 < settings.fetch.max_duration_s < settings.fetch.min_duration_s:
        raise SettingsError("fetch.max_duration_s must not be below min_duration_s")
    if settings.batch.notify_when not in {"always", "failure", "never"}:
        raise SettingsError("batch.notify_when must be always, failure or never")
    if settings.batch.notify_webhook and not settings.batch.notify_webhook.startswith(
        ("http://", "https://")
    ):
        raise SettingsError("batch.notify_webhook must be an http(s) URL")
    _non_negative("batch.min_free_disk_gb", settings.batch.min_free_disk_gb)
    for name in ("cut_lead_s", "max_switches", "switch_window_s", "max_shot_s"):
        _non_negative(f"program.{name}", getattr(settings.program, name))
    _positive("program.wide_insert_s", settings.program.wide_insert_s)
    if settings.program.cut_lead_s > 2:
        raise SettingsError("program.cut_lead_s must not exceed 2")
    if 0 < settings.program.max_shot_s <= settings.program.wide_insert_s:
        raise SettingsError("program.max_shot_s must exceed wide_insert_s when enabled")
    if settings.program.crossfade_ms > 200:
        raise SettingsError("program.crossfade_ms must not exceed 200")
    if any(
        not key.strip() or not value.strip()
        for key, value in settings.program.speaker_cameras.items()
    ):
        raise SettingsError("program.speaker_cameras requires nonempty speaker and camera names")
    if settings.program.fps != "auto":
        from fractions import Fraction

        try:
            fps = Fraction(settings.program.fps)
            if not 1 <= fps <= 240:
                raise ValueError("fps out of range")
        except (ValueError, ZeroDivisionError) as exc:
            raise SettingsError("program.fps must be auto or a rate between 1 and 240") from exc
    _positive("roughcut.pause_min_s", settings.roughcut.pause_min_s)
    _positive("roughcut.unknown_sound_min_s", settings.roughcut.unknown_sound_min_s)
    _non_negative("roughcut.pause_keep_s", settings.roughcut.pause_keep_s)
    _non_negative("roughcut.head_tail_pad_s", settings.roughcut.head_tail_pad_s)
    _non_negative("roughcut.retake_max_gap_s", settings.roughcut.retake_max_gap_s)
    _positive("roughcut.phrase_gap_threshold", settings.roughcut.phrase_gap_threshold)
    _one_of("text.prompt_language", settings.text.prompt_language, {"auto", "ru", "en"})
    if not settings.text.base_url.strip():
        raise SettingsError("text.base_url must not be empty")
    if not settings.text.model.strip():
        raise SettingsError("text.model must not be empty")
    for name in ("max_chars_chunk", "article_max_chars_chunk"):
        value = getattr(settings.text, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 500:
            raise SettingsError(f"text.{name} must be an integer of at least 500")
    if not isinstance(settings.text.min_similarity, (int, float)) or not (
        0 <= settings.text.min_similarity <= 1
    ):
        raise SettingsError("text.min_similarity must be between 0 and 1")
    _one_of("reels.prompt_language", settings.reels.prompt_language, {"auto", "ru", "en"})
    _one_of("reels.framing", settings.reels.framing, {"source", "crop", "fit"})
    _one_of("reels.tracking_device", settings.reels.tracking_device, {"cpu", "cuda"})
    if settings.reels.tracking and settings.reels.framing != "crop":
        raise SettingsError("reels.tracking requires reels.framing=crop")
    for name in ("width", "height"):
        value = getattr(settings.reels, name)
        if not 64 <= value <= 4096 or value % 2:
            raise SettingsError(f"reels.{name} must be an even integer between 64 and 4096")
    if not math.isfinite(settings.reels.crop_x) or not 0 <= settings.reels.crop_x <= 1:
        raise SettingsError("reels.crop_x must be between 0 and 1")
    if not settings.reels.base_url.strip() or not settings.reels.model.strip():
        raise SettingsError("reels.base_url and reels.model must not be empty")
    for name in (
        "chunk_seconds",
        "max_chars_chunk",
        "max_candidates",
        "review_batch_size",
        "cleanup_max_candidates",
        "judge_max_candidates",
    ):
        value = getattr(settings.reels, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SettingsError(f"reels.{name} must be a positive integer")
    if (
        not isinstance(settings.reels.audio_max_candidates, int)
        or isinstance(settings.reels.audio_max_candidates, bool)
        or settings.reels.audio_max_candidates < 1
    ):
        raise SettingsError("reels.audio_max_candidates must be a positive integer")
    for name in ("audio_silence_min_s", "audio_timeout_s"):
        _positive(f"reels.{name}", getattr(settings.reels, name))
    if (
        not math.isfinite(settings.reels.audio_noise_db)
        or not -100 <= settings.reels.audio_noise_db <= 0
    ):
        raise SettingsError("reels.audio_noise_db must be between -100 and 0")
    for name in ("json_retries", "retry_budget"):
        value = getattr(settings.reels, name)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
            raise SettingsError(f"reels.{name} must be an integer between 0 and 100")
    _positive("reels.request_timeout_s", settings.reels.request_timeout_s)
    for role, config in settings.llm.roles.items():
        if (
            role not in {
                "proofread", "article", "context", "scout", "cleanup", "judge", "roughcut",
            }
            or not isinstance(config, dict)
            or not config
            or set(config) - {"model", "base_url"}
            or any(not isinstance(v, str) or not v.strip() for v in config.values())
        ):
            raise SettingsError("invalid llm.roles model/base_url mapping")
        if settings.llm.managed and "base_url" in config:
            raise SettingsError("managed roles cannot override base_url")
    if any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in settings.text.term_fixes.items()
    ):
        raise SettingsError("invalid text.term_fixes")
    if settings.text.term_check_network and not settings.text.term_check:
        raise SettingsError("term_check_network requires term_check")
    if settings.text.term_check and not settings.text.proofread:
        raise SettingsError("term_check requires proofread")
    if (
        not isinstance(settings.text.term_max_candidates, int)
        or isinstance(settings.text.term_max_candidates, bool)
        or not 1 <= settings.text.term_max_candidates <= 100
    ):
        raise SettingsError("invalid text.term_max_candidates")
    _non_negative("reels.retry_backoff_s", settings.reels.retry_backoff_s)
    if settings.reels.retry_backoff_s > 30:
        raise SettingsError("reels.retry_backoff_s must not exceed 30 seconds")
    for name in ("target_min_s", "target_max_s"):
        _positive(f"reels.{name}", getattr(settings.reels, name))
    if settings.reels.target_max_s < settings.reels.target_min_s:
        raise SettingsError("reels.target_max_s must not be below reels.target_min_s")
    if (
        not math.isfinite(settings.reels.min_quote_ratio)
        or not 0 <= settings.reels.min_quote_ratio <= 1
    ):
        raise SettingsError("reels.min_quote_ratio must be between 0 and 1")
    from fractions import Fraction

    try:
        if not 1 <= Fraction(settings.reels.render_fps) <= 240:
            raise ValueError("fps out of range")
    except (ValueError, ZeroDivisionError) as exc:
        raise SettingsError("reels.render_fps must be a rate between 1 and 240") from exc
    if settings.roughcut.retake_min_words < 1:
        raise SettingsError("roughcut.retake_min_words must be a positive integer")
    threshold = settings.roughcut.silence_threshold_db
    if (
        not isinstance(threshold, (float, int))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not -100 <= threshold <= 0
    ):
        raise SettingsError("roughcut.silence_threshold_db must be between -100 and 0")
    if settings.roughcut.pause_keep_s >= settings.roughcut.pause_min_s:
        raise SettingsError("roughcut.pause_keep_s must be less than pause_min_s")
    unknown = set(settings.export.targets) - {"fcpxml", "xmeml", "multicam", "otio"}
    if unknown:
        raise SettingsError(f"export.targets contains unsupported values: {sorted(unknown)}")


def _validate_roughcut_llm(roughcut: RoughcutSettings) -> None:
    unknown = set(roughcut.llm_detectors) - {"start", "slips", "fillers", "delivery"}
    if unknown or len(set(roughcut.llm_detectors)) != len(roughcut.llm_detectors):
        raise SettingsError(
            "roughcut.llm_detectors must be unique values of start, slips, fillers, delivery"
        )
    _one_of("roughcut.llm_apply", roughcut.llm_apply, {"suggest", "cut"})
    _one_of("roughcut.llm_prompt_language", roughcut.llm_prompt_language, {"auto", "ru", "en"})
    if not math.isfinite(roughcut.llm_min_confidence) or not (
        0 <= roughcut.llm_min_confidence <= 1
    ):
        raise SettingsError("roughcut.llm_min_confidence must be between 0 and 1")
    for name in ("llm_max_cut_s", "llm_max_start_s", "request_timeout_s"):
        _positive(f"roughcut.{name}", getattr(roughcut, name))
    if roughcut.llm_chunk_chars < 500:
        raise SettingsError("roughcut.llm_chunk_chars must be an integer of at least 500")
    for name in ("llm_json_retries", "llm_retry_budget"):
        if not 0 <= getattr(roughcut, name) <= 100:
            raise SettingsError(f"roughcut.{name} must be an integer between 0 and 100")
    if not roughcut.base_url.strip() or not roughcut.model.strip():
        raise SettingsError("roughcut.base_url and roughcut.model must not be empty")


def _construct(data: dict[str, Any]) -> Settings:
    allowed = {field.name for field in Settings.__dataclass_fields__.values()}
    unknown = set(data) - allowed
    if unknown:
        raise SettingsError(f"unknown settings sections: {sorted(unknown)}")
    try:
        return Settings(
            llm=LlmSettings(**_section(data, "llm")),
            scan=ScanSettings(**_section(data, "scan")),
            sync=SyncSettings(**_section(data, "sync")),
            transcribe=TranscribeSettings(**_section(data, "transcribe")),
            roughcut=RoughcutSettings(**_section(data, "roughcut")),
            export=ExportSettings(**_section(data, "export")),
            program=ProgramSettings(**_section(data, "program")),
            compute=ComputeSettings(**_section(data, "compute")),
            speakers=SpeakerSettings(**_section(data, "speakers")),
            text=TextSettings(**_section(data, "text")),
            reels=ReelsSettings(**_section(data, "reels")),
            fetch=FetchSettings(**_section(data, "fetch")),
            batch=BatchSettings(**_section(data, "batch")),
        )
    except TypeError as exc:
        raise SettingsError(str(exc)) from exc


def _matches_type(value: Any, expected: Any) -> bool:
    origin = get_origin(expected)
    if origin in (Union, types.UnionType):
        return any(_matches_type(value, item) for item in get_args(expected))
    if origin is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(expected)[0]) for item in value
        )
    if origin is dict:
        key_type, value_type = get_args(expected)
        return isinstance(value, dict) and all(
            _matches_type(key, key_type) and _matches_type(item, value_type)
            for key, item in value.items()
        )
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise SettingsError(f"{name} must be a mapping")
    return value


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _set_override(settings: dict[str, Any], expression: str) -> None:
    if "=" not in expression or "." not in expression.split("=", 1)[0]:
        raise SettingsError(f"invalid --set expression: {expression}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    target = settings
    for key in keys[:-1]:
        value = target.setdefault(key, {})
        if not isinstance(value, dict):
            raise SettingsError(f"cannot set child of scalar setting: {key}")
        target = value
    try:
        target[keys[-1]] = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise SettingsError(f"invalid override value: {raw}") from exc


def _one_of(name: str, value: str, choices: set[str]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise SettingsError(f"{name} must be one of {sorted(choices)}")


def _positive(name: str, value: float) -> None:
    valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_number or not math.isfinite(value) or value <= 0:
        raise SettingsError(f"{name} must be a finite positive number")


def _non_negative(name: str, value: float) -> None:
    valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_number or not math.isfinite(value) or value < 0:
        raise SettingsError(f"{name} must be a finite non-negative number")
