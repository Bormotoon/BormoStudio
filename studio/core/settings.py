"""Typed, layered Studio settings with fail-fast validation."""

from __future__ import annotations

import math
import re
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
    # "cached" reuses source/artifact digests while (size, mtime, ctime, inode)
    # and a content sample match; "always" rehashes every file once per run.
    verify_digests: str = "cached"
    # Compare the channels of multichannel audio (dual mono, safety copy, dead input).
    channel_analysis: str = "auto"
    # Files shorter than this are false starts (a camera switched on and off by
    # mistake) and are ignored; 0 keeps everything.
    min_duration_s: float = 3.0
    # Media suffixes beyond .mp4 .mov .mkv .avi .wav .flac .mp3 .m4a (".mts").
    extra_extensions: list[str] = field(default_factory=list)


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
    # Acoustic grid (verdict, plan refinement, drift check): window spacing and length.
    verify: bool = True
    refine_grid_s: float = 2.5
    refine_window_s: float = 4.0
    # "auto" refines the plan acoustically when the camera has sound.
    acoustic_refine: str = "auto"
    step_min_ms: float = 10.0
    # locate: search other cameras of a group this far around their predicted place.
    locate_search_s: float = 120.0
    # "text": every camera is transcribed and placed by its speech first (the
    # WhisperSync way), the acoustics then refine around that place;
    # "acoustic": locate searches the recorders first, text only for the rest.
    placement: str = "text"
    # A clip neither text nor acoustics placed is transcribed again in quality.
    rescue: str = "quality"
    # Acoustic grid of the drift check and refinement: largest lag it can see.
    # Recorder jumps of 0.2-0.8 s (sony_run) are invisible at 0.25 s.
    refine_max_lag_s: float = 1.0
    # A recorder that loses samples (a phone dropping buffers) is repaired
    # against the cameras: "auto" does it when losses are found, "off" never.
    recorder_repair: str = "auto"
    # What fills the lost time: the camera that heard it, or silence.
    recorder_fill: str = "camera"
    voice_segment_minutes: float = 0.0
    master_wav: bool = False
    master_crossfade_ms: float = 10.0
    self_check: str = "off"
    voice_enhance: str = "off"
    ambience: bool = False
    ambience_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    # Tempo conform of rendered pieces: auto (resample small factors, atempo
    # large ones), resample or atempo; seams snap to recorder silence within
    # seam_snap_max_s; the sync self-check transcribes in this Whisper mode
    # ("": transcribe.mode).
    stretch_method: str = "auto"
    seam_snap_max_s: float = 0.4
    self_check_transcribe_mode: str = ""
    # Lip-sync calibration (WhisperSync's): milliseconds a camera's picture is
    # moved against all sound, for every camera and per camera (asset id, group,
    # sub-folder or file name). Positive delays the sound (it arrived early).
    camera_av_offset_ms: float = 0.0
    camera_av_offset_ms_by_camera: dict[str, float] = field(default_factory=dict)
    # Pause ducking (WhisperSync's, off by default): rendered voices are lowered by
    # pause_duck_db where neither the camera nor the recorder has words for at
    # least pause_duck_min_pause_s, hiding room-tone mismatch in the gaps.
    pause_duck_enabled: bool = False
    pause_duck_db: float = -18.0
    pause_duck_fade_ms: int = 80
    pause_duck_min_pause_s: float = 0.6


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
    # Whisper clips merge VAD fragments only across pauses this short.
    vad_merge_gap_s: float = 1.5
    # Segments split at longer word gaps; 1-2 word series beyond them are checked.
    max_word_gap_s: float = 2.0
    # "auto"/"cameras" transcribe a recorder only around located cameras (whole
    # file when they cover most of it); "full" always transcribes everything.
    recorder_scope: str = "auto"
    window_margin_s: float = 60.0
    # Manual recorder ranges in seconds: {asset_id: [[start, end], ...]}.
    ranges: dict[str, list[list[float]]] = field(default_factory=dict)
    # "auto": camera transcripts only where needed (not located, self-check,
    # camera mode, no recorder); "always": every camera.
    camera_transcripts: str = "auto"
    # Whisper mode of the camera transcripts that only place a camera on the
    # recorder; the content transcripts follow ``mode``.
    camera_mode: str = "fast"
    # Decoding: candidates sampled at a fallback temperature, beam patience,
    # and whether each window sees the previous text (null: quality mode only).
    best_of: int = 5
    patience: float = 1.0
    condition_on_previous_text: bool | None = None


@dataclass(slots=True)
class RoughcutSettings:
    enabled: bool = True
    mode: str = "cut"
    pause_min_s: float = 1.0
    pause_keep_s: float = 0.4
    silence_threshold_db: float = -40.0
    unknown_sound_min_s: float = 0.3
    head_tail_pad_s: float = 0.2
    # Head/tail before the first and after the last word that hold untranscribed
    # sound: "cut" (applied, confidence 0.7), "suggest" (not applied) or "keep".
    head_tail_sound: str = "cut"
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
    targets: list[str] = field(default_factory=lambda: ["multicam", "xmeml"])
    fcpxml_version: str = "1.9"
    # Which rough-cut markers reach the NLE: cuts, all (adds filler and
    # untranscribed-sound review hints) or none.
    markers: str = "cuts"
    # Cuts shorter than this get no marker (0 marks every accepted cut).
    marker_min_s: float = 0.0
    # Final Cut event, project and multicam name; empty: the material folder's name.
    name: str = ""


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
    # Processes for CPU-heavy steps (locating cameras); 0 = all cores but one.
    cpu_workers: int = 0
    # Free RAM held by libvirt VMs while a run (or a whole queue) lasts: shrink
    # vm_domains (empty: every running one) by balloon down to their use plus
    # vm_headroom_mb (never below vm_min_mb), and shut vm_stop_domains down
    # (GPU-passthrough VMs, whose memory a balloon cannot free); all given back
    # at the end.
    host_memory: bool = False
    vm_domains: list[str] = field(default_factory=list)
    vm_stop_domains: list[str] = field(default_factory=list)
    vm_shutdown_timeout_s: int = 180
    vm_headroom_mb: int = 1536
    vm_min_mb: int = 1024
    vm_connect: str = "qemu:///system"


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
    # How many people talk, when known (pyannote); 0 lets it decide.
    num_speakers: int = 0
    # One name per person across recorder files / diarized sources that do not
    # play together (separate sessions); thresholds are cosine similarities.
    match_files: bool = True
    match_threshold: float = 0.9
    embedding_threshold: float = 0.5


@dataclass(slots=True)
class TextSettings:
    proofread: bool = False
    article: bool = False
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    prompt_language: str = "auto"
    max_chars_chunk: int = 4000
    article_max_chars_chunk: int = 6000
    # Transcript window per article request (seconds; 0: characters only).
    article_chunk_seconds: float = 600.0
    # An article chunk is flagged as rewritten, not edited, beyond these: share
    # of words not in its source, length against the source (most and least)
    # and share of the source's words that must survive.
    article_max_novel_word_ratio: float = 0.15
    article_max_length_ratio: float = 1.15
    article_min_length_ratio: float = 0.25
    article_min_source_coverage: float = 0.45
    min_similarity: float = 0.8
    term_check: bool = False
    term_check_network: bool = False
    term_fixes: dict[str, str] = field(default_factory=dict)
    term_max_candidates: int = 10
    # A word is a suspect term from this many occurrences; pause between lookups.
    term_min_occurrences: int = 2
    term_pause_s: float = 1.0


@dataclass(slots=True)
class ReelsSettings:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local"
    prompt_language: str = "auto"
    # Prompt variant: chunk_<variant>.txt and friends, else the default ones.
    prompt_variant: str = "default"
    # Sample every answer under its JSON schema (a simpler one, then none, when
    # the server cannot compile it), so the JSON parses the first time.
    strict_json_schema: bool = True
    chunk_seconds: int = 600
    max_chars_chunk: int = 6000
    # Overlap of neighbouring scout chunks; -1: chunk_seconds/8 within 20-45 s.
    chunk_overlap_s: float = -1.0
    # How far a scout candidate may leave its chunk before it counts as invented.
    chunk_tolerance_s: float = 3.0
    # Clips per hour of runtime; the counts below then only set the type mix
    # (0: the counts are absolute).
    clips_per_hour: float = 10.0
    story_count: int = 2
    story_max_s: float = 15.0
    reel_count: int = 3
    reel_max_s: float = 60.0
    long_reel_count: int = 1
    long_reel_max_s: float = 180.0
    highlight_count: int = 5
    # Reel count when reel_count is 0, and the lengths asked for without types.
    max_candidates: int = 10
    target_min_s: float = 30.0
    target_max_s: float = 60.0
    # Quality filters applied at selection (0 turns one off): the judge's
    # 1-10 score and the clip length.
    min_score: float = 7.0
    min_duration_s: float = 15.0
    max_duration_s: float = 180.0
    cleanup: bool = True
    judge: bool = True
    # Candidates per cleanup request; also the floor of the cleanup input cap.
    review_batch_size: int = 16
    judge_batch_size: int = 14
    # Optional caps on the cleanup and judge input (0: Forge's own sizing).
    cleanup_max_candidates: int = 0
    judge_max_candidates: int = 0
    # Drop cleanup/judge outputs that no input candidate explains.
    require_candidate_overlap: bool = True
    # The judge sees the real first and last words of each clip.
    judge_context: bool = True
    judge_head_s: float = 15.0
    judge_tail_s: float = 5.0
    judge_excerpt_chars: int = 260
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
    context_digest_chars: int = 4000
    # Quote match below scout_quote_ratio is rejected before review, below
    # min_quote_ratio before selection; quote_refine widens a clip to its quote.
    scout_quote_ratio: float = 0.55
    min_quote_ratio: float = 0.75
    quote_refine: bool = True
    # Pull clip edges to phrase boundaries, at most this far.
    snap_boundaries: bool = True
    snap_max_shift_s: float = 3.0
    # Ranking factor weights (base, hook, readability, completeness, speaker,
    # duration, mid_thought, quote, audio, speech_rate, replay); {} keeps defaults.
    scoring_weights: dict[str, float] = field(default_factory=dict)
    # MMR selection: quality minus mmr_lambda x similarity to chosen clips; a
    # clip at max_topic_similarity or above only fills an otherwise empty slot.
    diversity: bool = True
    max_topic_similarity: float = 0.5
    mmr_lambda: float = 0.7
    # Two final clips may share at most this share of the shorter one.
    max_overlap_ratio: float = 0.2
    render: bool = False
    render_fps: str = "25"
    burn_subtitles: bool = False
    # Reel edges follow the speech: silence kept before the first and after the
    # last word, distance kept from a neighbouring word, how far an edge may
    # grow to its sentence's edge and the longest fragment it drops instead.
    edge_lead_in_s: float = 0.6
    edge_tail_s: float = 0.3
    edge_guard_s: float = 0.1
    edge_max_extend_s: float = 4.0
    edge_max_trim_s: float = 1.5
    # Moments failing the quality filters at render time are listed in
    # rejected.json; render_rejected also encodes them into rejected/.
    render_rejected: bool = False
    # Check every reel (streams, duration; blackdetect decodes it all).
    qa: bool = True
    qa_blackdetect: bool = False
    export_webm: bool = False
    export_gif: bool = False
    export_audio: bool = False
    # All finished reels back to back in reels-preview.mp4.
    preview_reel: bool = True
    # Reel encodes: auto (NVENC if it works, else libx264), nvenc, libx264, ...
    encoder: str = "auto"
    video_bitrate: str = "8M"
    audio_bitrate: str = "192k"
    preset: str = "fast"
    nvenc_cq: int = 21
    nvenc_preset: str = "p5"
    framing: str = "source"
    width: int = 1080
    height: int = 1920
    crop_x: float = 0.5
    tracking: bool = False
    tracking_device: str = "cuda"
    active_speaker: bool = True
    # Face tracking: faces below this height (source px) are ignored; follow
    # the face within a turn; two steadily visible people stacked ("split") or
    # the speaker alone; a new speaker is a cut or, when close, a pan.
    face_min_size: int = 40
    face_follow: bool = True
    two_speaker_layout: str = "speaker"
    speaker_switch: str = "cut"
    # With tracking, shots without a face (screenshots, slides, wide inserts):
    # "fit" shows them whole over a blurred fill, "crop" cuts the centre out.
    faceless_shots: str = "fit"
    # With tracking, a reel showing a face in less than this share of its
    # samples is rejected (0 keeps every reel).
    face_min_ratio: float = 0.3
    # Integrated loudness target for rendered reels (0 keeps source loudness).
    loudness_lufs: float = -14.0
    covers: bool = True
    # Rank with the YouTube "most replayed" heatmap of fetched episodes.
    replay: bool = True
    # Timeline reels switch to each speaker's camera (program camera policy).
    speaker_angles: bool = True
    # "video" keeps the picture; "title" adds a top band with the hook/title;
    # "audiogram" draws a waveform under the title (works without cameras).
    layout: str = "video"
    title_field: str = "hook"
    title_band: float = 0.22
    title_seconds: float = 0.0
    title_font: str = "Montserrat Black"
    title_color: str = "#FFFFFF"
    background_color: str = "#111114"
    waveform_color: str = "#FFFFFF"
    # Forge caption looks (subtitle-presets.js) for reels without manual edits;
    # "" keeps the editor's plain style. Empty/-1/0 keep the preset's choice.
    caption_preset: str = ""
    caption_highlight: str = ""
    caption_case: str = ""
    caption_punctuation: str = ""
    caption_max_words: int = -1
    caption_max_lines: int = 0
    caption_max_chars: int = 0
    caption_width_ratio: float = 0.74
    caption_font_size: int = 96
    caption_highlight_color: str = ""
    caption_pause_split_s: float = 0.5
    caption_max_duration_s: float = 7.0
    caption_min_gap_s: float = 0.15
    caption_fade_min_gap_s: float = 0.3
    caption_censor: list[str] = field(default_factory=list)
    caption_censor_style: str = "middle"
    # Styled captions: a new cue at each change of speaker, colours by speaker
    # in order of appearance (#RRGGBB), and the vertical placement.
    caption_split_on_speaker: bool = True
    caption_speaker_colors: list[str] = field(default_factory=list)
    caption_vertical_align: str = "style"
    caption_vertical_offset: float = 0.0
    # Listen to each reel again with Whisper and take its word timings when the
    # transcript's drifted (p95 of word ends at least caption_sync_threshold_s)
    # and enough words matched; "" model: transcribe.model.
    caption_whisper_sync: bool = False
    caption_sync_model: str = ""
    caption_sync_min_match: float = 0.5
    caption_sync_threshold_s: float = 0.2


@dataclass(slots=True)
class LlmSettings:
    managed: bool = False
    executable: str = "llama-server"
    model_path: str = ""
    port: int = 8080
    startup_timeout_s: float = 120.0
    roles: dict[str, dict[str, str]] = field(default_factory=dict)
    # Owned llama-server tuning; None leaves the server's own default.
    n_gpu_layers: int | None = None
    ctx_size: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    threads: int | None = None
    main_gpu: int | None = None
    # One slot: Studio sends requests one at a time, extra slots only cost VRAM.
    parallel: int | None = 1
    flash_attn: str = "auto"
    cache_type_k: str = ""
    cache_type_v: str = ""
    # Host-RAM prompt cache in MiB, "auto" (free memory minus the reserve) or None.
    cache_ram_mb: int | str | None = None
    cache_ram_reserve_mb: int = 6144
    cache_ram_min_mb: int = 512
    cache_ram_max_mb: int = 8192
    ctx_checkpoints: int | None = None
    log_file: str = "llama-server.log"
    extra_args: list[str] = field(default_factory=list)
    # Request options; role_options overrides them per task role.
    temperature: float = 0.0
    n_predict: int | None = None
    role_options: dict[str, dict[str, float]] = field(default_factory=dict)
    # Streamed responses are abandoned and retried when they stall.
    watchdog: bool = True
    first_token_timeout_s: float = 120.0
    stall_timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 2.0


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


# Settings added after projects were processed. At these values they stay out
# of stage fingerprints (see fingerprint_view), so adding one recomputes nothing.
LATER_DEFAULTS: dict[str, Any] = {
    "scan.extra_extensions": [],
    "sync.stretch_method": "auto",
    "sync.seam_snap_max_s": 0.4,
    "sync.self_check_transcribe_mode": "",
    "sync.camera_av_offset_ms": 0.0,
    "sync.camera_av_offset_ms_by_camera": {},
    "sync.pause_duck_enabled": False,
    "sync.pause_duck_db": -18.0,
    "sync.pause_duck_fade_ms": 80,
    "sync.pause_duck_min_pause_s": 0.6,
    "reels.faceless_shots": "fit",
}
# Kept under a section whose stage does not read them. The lip-sync calibration
# only moves pictures in exports, the program, previews and reels, which
# fingerprint it themselves (av_calibration.calibration_view); changing it must
# not synchronize again.
NOT_FINGERPRINTED = frozenset({"sync.camera_av_offset_ms", "sync.camera_av_offset_ms_by_camera"})
_UNSET = object()


def fingerprint_view(section: str, values: Any) -> Any:
    """``values`` of a settings section without later settings at their defaults,
    the keys in NOT_FINGERPRINTED and the pause-ducking options while it is off."""
    if not isinstance(values, dict):
        return values
    ducking_off = section == "sync" and not values.get("pause_duck_enabled", False)
    return {key: value for key, value in values.items()
            if LATER_DEFAULTS.get(f"{section}.{key}", _UNSET) != value
            and f"{section}.{key}" not in NOT_FINGERPRINTED
            and not (ducking_off and key.startswith("pause_duck_"))}


# Allowed values of the enumerated text settings; the desktop form offers them.
CHOICES: dict[str, tuple[str, ...]] = {
    "scan.grouping": ("auto", "folders", "names"),
    "scan.verify_digests": ("cached", "always"),
    "scan.channel_analysis": ("auto", "off"),
    "sync.mode": ("auto", "camera", "simple", "complex"),
    "sync.recorder_mode": ("best", "all"),
    "sync.acoustic_refine": ("auto", "off"),
    "sync.placement": ("text", "acoustic"),
    "sync.rescue": ("quality", "off"),
    "sync.recorder_repair": ("auto", "off"),
    "sync.recorder_fill": ("camera", "silence"),
    "sync.stretch_method": ("auto", "resample", "atempo"),
    "sync.self_check_transcribe_mode": ("", "fast", "quality"),
    "sync.self_check": ("off", "warn", "repair"),
    "sync.voice_enhance": ("off", "denoise", "denoise_dereverb", "resemble"),
    "transcribe.device": ("auto", "cuda", "cpu", "mlx"),
    "transcribe.mode": ("fast", "quality"),
    "transcribe.recorder_scope": ("auto", "cameras", "full"),
    "transcribe.camera_transcripts": ("auto", "always"),
    "transcribe.camera_mode": ("fast", "quality"),
    "roughcut.mode": ("cut", "markers"),
    "roughcut.head_tail_sound": ("cut", "suggest", "keep"),
    "roughcut.llm_apply": ("suggest", "cut"),
    "roughcut.llm_prompt_language": ("auto", "ru", "en"),
    "export.markers": ("cuts", "all", "none"),
    "program.encoder": ("auto", "nvenc", "videotoolbox", "libx264", "cpu"),
    "speakers.method": ("auto", "mics", "pyannote", "off"),
    "speakers.device": ("cpu", "cuda"),
    "text.prompt_language": ("auto", "ru", "en"),
    "reels.prompt_language": ("auto", "ru", "en"),
    "reels.framing": ("source", "crop", "fit"),
    "reels.encoder": ("auto", "nvenc", "videotoolbox", "libx264", "cpu"),
    "reels.caption_vertical_align": ("style", "top", "center", "bottom"),
    "reels.two_speaker_layout": ("speaker", "split"),
    "reels.speaker_switch": ("cut", "pan"),
    "reels.faceless_shots": ("fit", "crop"),
    "reels.tracking_device": ("cpu", "cuda"),
    "reels.layout": ("video", "title", "audiogram"),
    "reels.title_field": ("hook", "title"),
    "llm.flash_attn": ("auto", "on", "off"),
    "fetch.download": ("video", "audio"),
    "batch.notify_when": ("always", "failure", "never"),
}


def validate_settings(settings: Settings) -> None:
    for section_name in Settings.__dataclass_fields__:
        section = getattr(settings, section_name)
        for name, expected in get_type_hints(type(section)).items():
            if not _matches_type(getattr(section, name), expected):
                raise SettingsError(f"{section_name}.{name} has an invalid value type")
    for dotted, choices in CHOICES.items():
        section_name, name = dotted.split(".")
        _one_of(dotted, getattr(getattr(settings, section_name), name), set(choices))
    _validate_llm_tuning(settings.llm)
    if not 1 <= settings.llm.port <= 65535:
        raise SettingsError("llm.port must be between 1 and 65535")
    _positive("llm.startup_timeout_s", settings.llm.startup_timeout_s)
    if settings.llm.managed and (
        not settings.llm.model_path.strip() or not settings.llm.executable.strip()
    ):
        raise SettingsError("managed llama requires executable and model_path")
    if settings.compute.vm_shutdown_timeout_s < 30:
        raise SettingsError("compute.vm_shutdown_timeout_s must be at least 30")
    if settings.compute.vm_headroom_mb < 0 or settings.compute.vm_min_mb < 64:
        raise SettingsError("compute.vm_headroom_mb must be >= 0 and vm_min_mb >= 64")
    if settings.compute.cpu_workers < 0:
        raise SettingsError("compute.cpu_workers must be 0 (auto) or a positive count")
    _positive("scan.probe_timeout_s", settings.scan.probe_timeout_s)
    _non_negative("scan.min_duration_s", settings.scan.min_duration_s)
    if any(not re.fullmatch(r"\.[a-z0-9]{1,8}", item) for item in settings.scan.extra_extensions):
        raise SettingsError("scan.extra_extensions must be lowercase suffixes such as .mts")
    _non_negative("sync.seam_snap_max_s", settings.sync.seam_snap_max_s)
    if settings.sync.strategy not in {1, 2, 3}:
        raise SettingsError("sync.strategy must be 1, 2, or 3")
    _non_negative("sync.max_drift_ms", settings.sync.max_drift_ms)
    _non_negative("sync.voice_segment_minutes", settings.sync.voice_segment_minutes)
    _non_negative("sync.master_crossfade_ms", settings.sync.master_crossfade_ms)
    if any(not key.strip() or not math.isfinite(value)
           for key, value in settings.sync.session_offsets.items()):
        raise SettingsError("sync.session_offsets requires names and finite seconds")
    if not settings.sync.reference_camera.strip():
        raise SettingsError("sync.reference_camera must not be empty")
    if settings.sync.self_check == "repair" and settings.sync.voice_enhance != "off":
        raise SettingsError("sync.self_check=repair currently requires voice_enhance=off")
    for name in ("refine_grid_s", "refine_window_s", "step_min_ms"):
        _positive(f"sync.{name}", getattr(settings.sync, name))
    for name in ("acoustic_grid_s", "acoustic_window_s", "acoustic_min_sharpness"):
        _positive(f"sync.{name}", getattr(settings.sync, name))
    _positive("transcribe.repetition_penalty", settings.transcribe.repetition_penalty)
    _positive("transcribe.patience", settings.transcribe.patience)
    if settings.transcribe.best_of < 1:
        raise SettingsError("transcribe.best_of must be at least 1")
    _positive("transcribe.vad_merge_gap_s", settings.transcribe.vad_merge_gap_s)
    _positive("transcribe.max_word_gap_s", settings.transcribe.max_word_gap_s)
    _non_negative("transcribe.window_margin_s", settings.transcribe.window_margin_s)
    _positive("sync.locate_search_s", settings.sync.locate_search_s)
    _positive("sync.refine_max_lag_s", settings.sync.refine_max_lag_s)
    offsets = {"": settings.sync.camera_av_offset_ms, **settings.sync.camera_av_offset_ms_by_camera}
    if any(not math.isfinite(value) or abs(value) > 2000 for value in offsets.values()) or any(
            not key.strip() for key in settings.sync.camera_av_offset_ms_by_camera):
        raise SettingsError("sync.camera_av_offset_ms needs camera names and finite "
                            "milliseconds within ±2000")
    if not math.isfinite(settings.sync.pause_duck_db) or settings.sync.pause_duck_db > 0:
        raise SettingsError("sync.pause_duck_db must be 0 dB or lower")
    _non_negative("sync.pause_duck_fade_ms", settings.sync.pause_duck_fade_ms)
    _positive("sync.pause_duck_min_pause_s", settings.sync.pause_duck_min_pause_s)
    for asset_id, spans in settings.transcribe.ranges.items():
        if not asset_id.strip() or any(
                len(span) != 2 or not 0 <= span[0] < span[1] for span in spans):
            raise SettingsError("transcribe.ranges maps assets to [start, end] seconds")
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
    _validate_roughcut_llm(settings.roughcut)
    _non_negative("speakers.margin_db", settings.speakers.margin_db)
    if settings.speakers.num_speakers < 0:
        raise SettingsError("speakers.num_speakers must be 0 (automatic) or a count")
    _positive("speakers.step_s", settings.speakers.step_s)
    _non_negative("speakers.min_turn_s", settings.speakers.min_turn_s)
    _non_negative("speakers.max_gap_s", settings.speakers.max_gap_s)
    for name in ("match_threshold", "embedding_threshold"):
        value = getattr(settings.speakers, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SettingsError(f"speakers.{name} must be between 0 and 1")
    if settings.speakers.step_s > 1:
        raise SettingsError("speakers.step_s must not exceed 1 second")
    if not math.isfinite(settings.speakers.silence_floor_db) or not (
        -100 <= settings.speakers.silence_floor_db <= 0
    ):
        raise SettingsError("speakers.silence_floor_db must be between -100 and 0")
    _non_negative("program.min_shot_s", settings.program.min_shot_s)
    _non_negative("program.crossfade_ms", settings.program.crossfade_ms)
    lufs = settings.reels.loudness_lufs
    if not isinstance(lufs, (int, float)) or isinstance(lufs, bool) or not (
        lufs == 0 or -70 <= lufs <= -5
    ):
        raise SettingsError("reels.loudness_lufs must be 0 (off) or between -70 and -5")
    for name in ("max_height", "retries", "min_duration_s", "max_duration_s"):
        _non_negative(f"fetch.{name}", getattr(settings.fetch, name))
    if 0 < settings.fetch.max_duration_s < settings.fetch.min_duration_s:
        raise SettingsError("fetch.max_duration_s must not be below min_duration_s")
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
    for name in ("article_chunk_seconds", "term_pause_s"):
        _non_negative(f"text.{name}", getattr(settings.text, name))
    for name in ("article_max_novel_word_ratio", "article_min_length_ratio",
                 "article_min_source_coverage"):
        value = getattr(settings.text, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SettingsError(f"text.{name} must be between 0 and 1")
    if not settings.text.article_max_length_ratio >= 1:
        raise SettingsError("text.article_max_length_ratio must be at least 1")
    if settings.text.term_min_occurrences < 1:
        raise SettingsError("text.term_min_occurrences must be at least 1")
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
    if settings.reels.tracking and settings.reels.framing != "crop":
        raise SettingsError("reels.tracking requires reels.framing=crop")
    for name in ("width", "height"):
        value = getattr(settings.reels, name)
        if not 64 <= value <= 4096 or value % 2:
            raise SettingsError(f"reels.{name} must be an even integer between 64 and 4096")
    if not math.isfinite(settings.reels.crop_x) or not 0 <= settings.reels.crop_x <= 1:
        raise SettingsError("reels.crop_x must be between 0 and 1")
    _validate_reel_layout(settings.reels)
    if not settings.reels.base_url.strip() or not settings.reels.model.strip():
        raise SettingsError("reels.base_url and reels.model must not be empty")
    for name in (
        "chunk_seconds",
        "max_chars_chunk",
        "max_candidates",
        "review_batch_size",
        "judge_batch_size",
        "judge_excerpt_chars",
        "context_digest_chars",
    ):
        value = getattr(settings.reels, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SettingsError(f"reels.{name} must be a positive integer")
    for name in ("cleanup_max_candidates", "judge_max_candidates", "story_count",
                 "reel_count", "long_reel_count", "highlight_count"):
        value = getattr(settings.reels, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SettingsError(f"reels.{name} must be 0 or a positive integer")
    for name in ("clips_per_hour", "story_max_s", "reel_max_s", "long_reel_max_s",
                 "min_score", "min_duration_s", "max_duration_s", "chunk_tolerance_s",
                 "judge_head_s", "judge_tail_s", "snap_max_shift_s"):
        _non_negative(f"reels.{name}", getattr(settings.reels, name))
    if settings.reels.chunk_overlap_s != -1 and not 0 <= settings.reels.chunk_overlap_s:
        raise SettingsError("reels.chunk_overlap_s must be -1 (auto) or non-negative")
    for name in ("scout_quote_ratio", "min_quote_ratio", "max_topic_similarity",
                 "mmr_lambda", "max_overlap_ratio"):
        value = getattr(settings.reels, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise SettingsError(f"reels.{name} must be between 0 and 1")
    for name in ("edge_lead_in_s", "edge_tail_s", "edge_guard_s", "edge_max_extend_s",
                 "edge_max_trim_s"):
        _non_negative(f"reels.{name}", getattr(settings.reels, name))
    for name in ("video_bitrate", "audio_bitrate"):
        if not re.fullmatch(r"\d+(\.\d+)?[kKmM]?", getattr(settings.reels, name)):
            raise SettingsError(f"reels.{name} must be a bitrate such as 8M or 192k")
    if not 0 <= settings.reels.nvenc_cq <= 51:
        raise SettingsError("reels.nvenc_cq must be between 0 and 51")
    if not re.fullmatch(r"p[1-7]", settings.reels.nvenc_preset):
        raise SettingsError("reels.nvenc_preset must be p1 ... p7")
    if settings.reels.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast",
                                     "medium", "slow", "slower", "veryslow"}:
        raise SettingsError("reels.preset must be an x264 preset such as fast")
    if any(not re.fullmatch(r"#[0-9a-fA-F]{6}", colour)
           for colour in settings.reels.caption_speaker_colors):
        raise SettingsError("reels.caption_speaker_colors must be #RRGGBB colours")
    if not 0 <= settings.reels.face_min_ratio <= 1:
        raise SettingsError("reels.face_min_ratio must be between 0 and 1")
    if settings.reels.face_min_size < 1:
        raise SettingsError("reels.face_min_size must be a positive number of pixels")
    if not 0 <= settings.reels.caption_sync_min_match <= 1:
        raise SettingsError("reels.caption_sync_min_match must be between 0 and 1")
    _non_negative("reels.caption_sync_threshold_s", settings.reels.caption_sync_threshold_s)
    if not -0.5 <= settings.reels.caption_vertical_offset <= 0.5:
        raise SettingsError("reels.caption_vertical_offset must be between -0.5 and 0.5")
    if settings.reels.scout_quote_ratio > settings.reels.min_quote_ratio:
        raise SettingsError("reels.scout_quote_ratio must not exceed reels.min_quote_ratio")
    if 0 < settings.reels.max_duration_s < settings.reels.min_duration_s:
        raise SettingsError("reels.max_duration_s must not be below min_duration_s")
    from studio.reels.analysis.scoring import DEFAULT_SCORING_WEIGHTS

    unknown_factors = set(settings.reels.scoring_weights) - set(DEFAULT_SCORING_WEIGHTS)
    if unknown_factors or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in settings.reels.scoring_weights.values()
    ):
        raise SettingsError(
            "reels.scoring_weights must map factor names "
            f"({', '.join(DEFAULT_SCORING_WEIGHTS)}) to numbers")
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
            role not in LLM_ROLES
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
    _non_negative("export.marker_min_s", settings.export.marker_min_s)
    if not isinstance(settings.export.name, str):
        raise SettingsError("export.name must be text")


LLM_ROLES = {"proofread", "article", "context", "scout", "cleanup", "judge", "roughcut"}


def _validate_llm_tuning(llm: LlmSettings) -> None:
    for name in ("ctx_size", "batch_size", "ubatch_size", "threads", "parallel",
                 "ctx_checkpoints", "n_predict"):
        value = getattr(llm, name)
        if value is not None and value < 1:
            raise SettingsError(f"llm.{name} must be a positive integer or null")
    for name in ("n_gpu_layers", "main_gpu"):
        value = getattr(llm, name)
        if value is not None and value < 0:
            raise SettingsError(f"llm.{name} must be non-negative or null")
    for name in ("cache_type_k", "cache_type_v"):
        if not re.fullmatch(r"[a-z0-9_]*", getattr(llm, name)):
            raise SettingsError(f"llm.{name} must be a llama.cpp cache type such as q8_0")
    if (llm.cache_type_k or llm.cache_type_v) and llm.flash_attn == "off":
        raise SettingsError("quantized llama KV cache requires llm.flash_attn auto or on")
    ram = llm.cache_ram_mb
    if isinstance(ram, str) and ram != "auto" or isinstance(ram, int) and ram < 0:
        raise SettingsError("llm.cache_ram_mb must be MiB, auto or null")
    for name in ("cache_ram_reserve_mb", "cache_ram_min_mb", "cache_ram_max_mb"):
        if getattr(llm, name) < 0:
            raise SettingsError(f"llm.{name} must be non-negative")
    if llm.cache_ram_max_mb < llm.cache_ram_min_mb:
        raise SettingsError("llm.cache_ram_max_mb must not be below cache_ram_min_mb")
    if any(not isinstance(item, str) or not item for item in llm.extra_args):
        raise SettingsError("llm.extra_args must be a list of nonempty strings")
    if not 0 <= llm.temperature <= 2:
        raise SettingsError("llm.temperature must be between 0 and 2")
    for name in ("first_token_timeout_s", "stall_timeout_s"):
        _positive(f"llm.{name}", getattr(llm, name))
    if not 0 <= llm.max_retries <= 10:
        raise SettingsError("llm.max_retries must be between 0 and 10")
    _non_negative("llm.retry_backoff_s", llm.retry_backoff_s)
    for role, options in llm.role_options.items():
        if role not in LLM_ROLES or set(options) - {"temperature", "n_predict", "timeout_s"}:
            raise SettingsError("llm.role_options maps roles to temperature/n_predict/timeout_s")
        if not 0 <= options.get("temperature", 0) <= 2:
            raise SettingsError("llm.role_options temperature must be between 0 and 2")
        if options.get("n_predict", 1) < 1 or options.get("timeout_s", 1) <= 0:
            raise SettingsError("llm.role_options n_predict and timeout_s must be positive")


def _validate_reel_layout(reels: ReelsSettings) -> None:
    if not math.isfinite(reels.title_band) or not 0.1 <= reels.title_band <= 0.5:
        raise SettingsError("reels.title_band must be between 0.1 and 0.5")
    _non_negative("reels.title_seconds", reels.title_seconds)
    if not reels.title_font.strip() or any(char in reels.title_font for char in ",\n\r"):
        raise SettingsError("reels.title_font must be a font family name")
    for name in ("title_color", "background_color", "waveform_color"):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", getattr(reels, name)):
            raise SettingsError(f"reels.{name} must be #RRGGBB")
    _validate_captions(reels)


def _validate_captions(reels: ReelsSettings) -> None:
    from studio.core import caption_styles as captions

    if reels.caption_preset and reels.caption_preset not in captions.preset_names():
        raise SettingsError(f"reels.caption_preset must be one of {captions.preset_names()}")
    for name, choices in (("caption_highlight", captions.HIGHLIGHTS),
                          ("caption_case", captions.CASES),
                          ("caption_punctuation", captions.PUNCTUATION)):
        value = getattr(reels, name)
        if value:
            _one_of(f"reels.{name}", value, choices)
    _one_of("reels.caption_censor_style", reels.caption_censor_style, captions.CENSOR_STYLES)
    if reels.caption_max_words < -1 or not 0 <= reels.caption_max_lines <= 4:
        raise SettingsError("reels.caption_max_words >= -1 and caption_max_lines 0..4")
    if reels.caption_max_chars < 0 or not 12 <= reels.caption_font_size <= 300:
        raise SettingsError("reels.caption_max_chars >= 0 and caption_font_size 12..300")
    if not 0.2 <= reels.caption_width_ratio <= 1:
        raise SettingsError("reels.caption_width_ratio must be between 0.2 and 1")
    if reels.caption_highlight_color and not re.fullmatch(
        r"#[0-9a-fA-F]{6}", reels.caption_highlight_color
    ):
        raise SettingsError("reels.caption_highlight_color must be #RRGGBB or empty")
    for name in ("caption_pause_split_s", "caption_max_duration_s"):
        _positive(f"reels.{name}", getattr(reels, name))
    for name in ("caption_min_gap_s", "caption_fade_min_gap_s"):
        _non_negative(f"reels.{name}", getattr(reels, name))
    if any(not isinstance(word, str) or not word.strip() for word in reels.caption_censor):
        raise SettingsError("reels.caption_censor must list nonempty words")


def _validate_roughcut_llm(roughcut: RoughcutSettings) -> None:
    unknown = set(roughcut.llm_detectors) - {"start", "slips", "fillers", "delivery"}
    if unknown or len(set(roughcut.llm_detectors)) != len(roughcut.llm_detectors):
        raise SettingsError(
            "roughcut.llm_detectors must be unique values of start, slips, fillers, delivery"
        )
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
