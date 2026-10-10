"""Shared Whisper engine and project transcription stage."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from platformdirs import user_cache_path

from studio.core.digest import file_digest
from studio.core.project import Project, stable_fingerprint
from studio.core.transcript import Segment, Transcript, Word, to_srt
from studio.core.transcript_sanity import (
    SEGMENTATION_VERSION,
    repair_detached_words,
    speech_clips,
    suspect_count,
)
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput

CACHE_SCHEMA = "studio/transcript-cache/3"
TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass(frozen=True, slots=True)
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
    glossary: tuple[str, ...] = ()
    hotwords: str = ""
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    keep_fillers: bool = True
    use_cache: bool = True
    vad_merge_gap_s: float = 1.5
    max_word_gap_s: float = 2.0

    @classmethod
    def from_mapping(cls, data: object) -> TranscribeSettings:
        values = data if isinstance(data, dict) else {}
        glossary = values.get("glossary", [])
        if not isinstance(glossary, list) or not all(isinstance(item, str) for item in glossary):
            raise ValueError("transcribe.glossary must be a list of strings")
        return cls(
            model=str(values.get("model", "large-v3")),
            language=str(values["language"]) if values.get("language") else None,
            device=str(values.get("device", "auto")),
            compute_type=str(values.get("compute_type", "auto")),
            mode=str(values.get("mode", "fast")),
            batch_size=int(values.get("batch_size", 16)),
            beam_size=int(values.get("beam_size", 5)),
            quality_beam_size=int(values.get("quality_beam_size", 10)),
            initial_prompt=str(values.get("initial_prompt", "")),
            glossary=tuple(glossary),
            hotwords=str(values.get("hotwords", "")),
            repetition_penalty=float(values.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(values.get("no_repeat_ngram_size", 3)),
            keep_fillers=bool(values.get("keep_fillers", True)),
            use_cache=bool(values.get("use_cache", True)),
            vad_merge_gap_s=float(values.get("vad_merge_gap_s", 1.5)),
            max_word_gap_s=float(values.get("max_word_gap_s", 2.0)),
        )

    def prompt(self) -> str:
        parts = [self.initial_prompt.strip()]
        if self.glossary:
            parts.append("Terms: " + ", ".join(self.glossary))
        if self.keep_fillers:
            parts.append("Preserve spoken fillers such as ээ, мм, ну, um, and uh.")
        return "\n".join(part for part in parts if part)


class WhisperBackend(Protocol):
    device: str
    compute_type: str

    def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript: ...


class FasterWhisperBackend:
    """One lazily loaded faster-whisper model for the complete stage queue."""

    def __init__(self, settings: TranscribeSettings) -> None:
        self.settings = settings
        self.device = _resolve_device(settings.device)
        self.compute_type = _resolve_compute_type(self.device, settings.compute_type)
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            from studio.modules.models import whisper_model_source

            self._model = WhisperModel(
                whisper_model_source(self.settings.model),
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript:
        current = settings
        while True:
            try:
                return self._transcribe_once(audio, current)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if current.mode == "fast" and current.batch_size > 1:
                    current = replace(current, batch_size=max(1, current.batch_size // 2))
                elif current.mode == "fast":
                    current = replace(current, mode="quality", quality_beam_size=1)
                else:
                    raise
                self.unload()

    def _transcribe_once(self, audio: Path, settings: TranscribeSettings) -> Transcript:
        model = self._ensure_model()
        samples, clips = self._speech_clips(audio, settings)
        if not clips:
            return Transcript(audio, settings.language or "", len(samples) / 16000, [],
                              settings.model, self.device, self.compute_type, settings.mode)
        kwargs: dict[str, Any] = {
            "language": settings.language,
            "word_timestamps": True,
            "vad_filter": False,
            "temperature": TEMPERATURES,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "initial_prompt": settings.prompt() or None,
            "hotwords": settings.hotwords or ", ".join(settings.glossary) or None,
            "repetition_penalty": settings.repetition_penalty,
            "no_repeat_ngram_size": settings.no_repeat_ngram_size,
        }
        if settings.mode == "quality":
            kwargs.update(
                beam_size=settings.quality_beam_size,
                condition_on_previous_text=True,
            )
            generated, info = model.transcribe(
                samples, clip_timestamps=[value for clip in clips for value in clip], **kwargs
            )
        else:
            from faster_whisper import BatchedInferencePipeline  # type: ignore[import-not-found]

            kwargs.update(beam_size=settings.beam_size, condition_on_previous_text=False)
            generated, info = BatchedInferencePipeline(model=model).transcribe(
                samples, batch_size=max(1, settings.batch_size),
                clip_timestamps=[{"start": start, "end": end} for start, end in clips],
                **kwargs,
            )
        segments = [
            Segment(
                float(segment.start),
                float(segment.end),
                tuple(
                    Word(
                        str(word.word).strip(),
                        float(word.start),
                        float(word.end),
                        float(word.probability),
                    )
                    for word in (segment.words or [])
                ),
                avg_logprob=_optional_float(getattr(segment, "avg_logprob", None)),
                text_override=str(segment.text).strip(),
            )
            for segment in generated
        ]
        return Transcript(
            source_audio=audio,
            language=str(getattr(info, "language", settings.language or "")),
            duration=float(getattr(info, "duration", 0.0)),
            segments=segments,
            model=settings.model,
            device=self.device,
            compute_type=self.compute_type,
            mode=settings.mode,
        )

    @staticmethod
    def _speech_clips(
        audio: Path, settings: TranscribeSettings
    ) -> tuple[Any, list[tuple[float, float]]]:
        """Decoded 16 kHz samples and Studio's own VAD clips (see ``speech_clips``).

        faster-whisper would glue fragments into 30 s chunks across any silence
        and give a phrase's first word the time of a noise burst minutes
        earlier. A clip is a real range of the file, so word times need no
        restoring.
        """
        from faster_whisper import decode_audio  # type: ignore[import-not-found]
        from faster_whisper.vad import (  # type: ignore[import-not-found]
            VadOptions,
            get_speech_timestamps,
        )

        samples = decode_audio(str(audio), sampling_rate=16000)
        fragments = get_speech_timestamps(
            samples, VadOptions(max_speech_duration_s=30, min_silence_duration_ms=160))
        return samples, speech_clips(
            [(item["start"] / 16000, item["end"] / 16000) for item in fragments],
            merge_gap_s=settings.vad_merge_gap_s,
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
        with contextlib.suppress(ImportError, RuntimeError):
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Whisper names → MLX community conversions; other values are a repo or folder.
MLX_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class MlxWhisperBackend:
    """Whisper on Apple Silicon through mlx-whisper (Metal, unified memory).

    mlx-whisper has no batched pipeline, VAD filter, hotwords or repetition
    controls; fast mode decodes sequentially and glossary terms stay in the prompt.
    """

    device = "mlx"
    compute_type = "float16"

    def __init__(self, settings: TranscribeSettings) -> None:
        self.settings = settings
        self.model = MLX_MODELS.get(settings.model, settings.model)

    def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript:
        import mlx_whisper  # type: ignore[import-not-found]

        quality = settings.mode == "quality"
        result = mlx_whisper.transcribe(
            str(audio),
            path_or_hf_repo=self.model,
            language=settings.language,
            word_timestamps=True,
            temperature=tuple(TEMPERATURES),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt=settings.prompt() or None,
            condition_on_previous_text=quality,
            verbose=None,
        )
        segments = [
            Segment(
                float(segment["start"]),
                float(segment["end"]),
                tuple(
                    Word(str(word["word"]).strip(), float(word["start"]), float(word["end"]),
                         float(word.get("probability", 1.0)))
                    for word in segment.get("words") or []
                ),
                avg_logprob=_optional_float(segment.get("avg_logprob")),
                text_override=str(segment.get("text", "")).strip(),
            )
            for segment in result.get("segments", [])
        ]
        from studio.core.media import probe

        return Transcript(
            source_audio=audio,
            language=str(result.get("language") or settings.language or ""),
            duration=float(probe(audio).duration_s or (segments[-1].end if segments else 0.0)),
            segments=segments,
            model=settings.model,
            device=self.device,
            compute_type=self.compute_type,
            mode=settings.mode,
        )

    def unload(self) -> None:
        """MLX frees the model with the Python objects; nothing is held here."""


def whisper_backend(settings: TranscribeSettings) -> WhisperBackend:
    if _resolve_device(settings.device) == "mlx":
        return MlxWhisperBackend(settings)
    return FasterWhisperBackend(settings)


class WhisperEngine:
    def __init__(
        self,
        settings: TranscribeSettings,
        backend: WhisperBackend | None = None,
        cache_dir: Path | None = None,
        digest_cache: Path | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or whisper_backend(settings)
        self.cache_dir = cache_dir or user_cache_path("studio") / "transcripts"
        self.digest_cache = digest_cache

    def transcribe(self, path: Path, stream_index: int | None = None) -> Transcript:
        cache_path = self.cache_dir / f"{self._key(path, stream_index)}.json"
        if self.settings.use_cache:
            cached = self._load(cache_path)
            if cached is not None:
                cached.source_audio = path
                return cached
        if stream_index is None:
            transcript = self._sane(self.backend.transcribe(path, self.settings), path)
        else:
            if (not isinstance(stream_index, int)
                    or isinstance(stream_index, bool) or stream_index < 0):
                raise ValueError("audio stream index must be a non-negative integer")
            with tempfile.TemporaryDirectory(prefix="uvf-transcribe-") as temporary:
                decoded = Path(temporary) / "selected.wav"
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                     "-map", f"0:{stream_index}", "-vn", "-ac", "1", "-ar", "16000",
                     "-c:a", "pcm_s16le", str(decoded)],
                    check=True, capture_output=True,
                )
                transcript = self._sane(self.backend.transcribe(decoded, self.settings), decoded)
        transcript.source_audio = path
        if self.settings.use_cache:
            self._save(cache_path, transcript)
        return transcript

    def _sane(self, transcript: Transcript, audio: Path) -> Transcript:
        """Repair words torn from their phrase (any backend); see ``transcript_sanity``."""
        return repair_detached_words(transcript, energy_probe(audio),
                                     max_word_gap_s=self.settings.max_word_gap_s)

    def _key(self, path: Path, stream_index: int | None) -> str:
        identity = _sha256(path, self.digest_cache)
        return stable_fingerprint(
            CACHE_SCHEMA,
            SEGMENTATION_VERSION,
            identity,
            stream_index,
            self.settings,
            self.backend.device,
            self.backend.compute_type,
        )

    @staticmethod
    def _load(path: Path) -> Transcript | None:
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            if wrapper.get("cache_schema") != CACHE_SCHEMA:
                return None
            return Transcript.from_dict(wrapper["transcript"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def _save(path: Path, transcript: Transcript) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"cache_schema": CACHE_SCHEMA, "transcript": transcript.to_dict()},
                    handle,
                    ensure_ascii=False,
                )
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise


class TranscribeStage:
    id = "transcribe"
    title = "Transcription"
    after: tuple[str, ...] = ("prepare",)
    # Settings that can change this stage's result; they alone enter its fingerprint.
    settings_keys: tuple[str, ...] = ("transcribe",)
    gpu = GpuUse.WHISPER

    def requirements(self, settings: dict[str, Any]) -> list[Requirement]:
        device = str(settings.get("transcribe", {}).get("device", "auto"))
        if device == "mlx" or device == "auto" and mlx_available():
            return [Requirement("package", "mlx-whisper", "speech recognition on Apple Silicon")]
        return [Requirement("package", "faster-whisper", "local speech recognition")]

    def decide(self, project: Project, settings: dict[str, Any]) -> Decision:
        requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
        if not requirements.is_file():
            return Decision.blocked("prepare requirements are missing", "run prepare")
        data = json.loads(requirements.read_text(encoding="utf-8"))
        return Decision.run() if data.get("transcripts") else Decision.skip(
            "no source requires transcription"
        )

    def fingerprint(self, project: Project, settings: dict[str, Any]) -> str:
        requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
        content = requirements.read_text(encoding="utf-8") if requirements.is_file() else ""
        requested = json.loads(content).get("transcripts", []) if content else []
        assets = {asset.id: asset for asset in project.assets}
        identities = {}
        for asset_id in requested:
            asset = assets.get(asset_id)
            source = project.source_dir / asset.path if asset else None
            identities[asset_id] = (
                {"path": str(source), "sha256": _sha256(source, project.work_dir)}
                if source is not None and source.is_file() else {"missing": True}
            )
        return stable_fingerprint(
            "transcribe-v4", SEGMENTATION_VERSION, content, identities,
            settings.get("transcribe", {}),
        )

    def explain_stale(
        self, project: Project, settings: dict[str, Any], manifest: Any
    ) -> str | None:
        requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
        if not requirements.is_file():
            return None
        requested = json.loads(requirements.read_text(encoding="utf-8")).get("transcripts", [])
        published = {Path(artifact.path).stem for artifact in manifest.artifacts}
        missing = [asset_id for asset_id in requested if asset_id not in published]
        return f"needs transcript {', '.join(missing)}" if missing else None

    def run(self, context: StageContext) -> StageOutput:
        requirements_path = context.project.work_dir / "stages" / "prepare" / "requirements.json"
        requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
        asset_by_id = {asset.id: asset for asset in context.project.assets}
        settings = TranscribeSettings.from_mapping(context.settings.get("transcribe", {}))
        engine = WhisperEngine(settings, digest_cache=context.project.work_dir)
        artifacts: list[Path] = []
        warnings: list[str] = []
        transcripts = dict(context.project.transcripts)
        for asset_id in requirements.get("transcripts", []):
            asset = asset_by_id[asset_id]
            source = context.project.source_dir / asset.path
            stream = requirements.get("audio_streams", {}).get(asset_id)
            transcript = engine.transcribe(source, stream)
            output = context.work_dir / "stages" / "transcribe" / f"{asset_id}.json"
            srt = output.with_suffix(".srt")
            transcript.save(output)
            srt.write_text(to_srt(transcript), encoding="utf-8")
            artifacts.extend((output, srt))
            transcripts[asset_id] = output
            if suspect_count(transcript):
                words = ", ".join(f"«{item['text']}» at {item['start']:.1f} s"
                                  for item in transcript.metadata["suspect_words"][:5])
                warnings.append(f"{asset.path}: words placed in silence, check them: {words}")
        return StageOutput(tuple(artifacts), {"transcripts": transcripts}, tuple(warnings))


def mlx_available() -> bool:
    import importlib.util
    import platform
    import sys

    return (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_whisper") is not None)


def energy_probe(audio: Path) -> Callable[[float, float], float]:
    """RMS level (dBFS) of ``[start, end]`` of ``audio``, decoded on demand."""

    def energy(start: float, end: float) -> float:
        start = max(0.0, start)
        if end <= start:
            return -120.0
        try:
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t",
                 f"{end - start:.3f}", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000",
                 "-f", "f32le", "-"],
                check=True, capture_output=True, timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return 0.0  # unknown: treat as sound, so nothing is marked silent
        import numpy as np

        samples = np.frombuffer(result.stdout, dtype=np.float32)
        if not samples.size:
            return -120.0
        return float(20 * np.log10(np.sqrt(np.mean(samples.astype(np.float64) ** 2)) + 1e-9))

    return energy


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2  # type: ignore[import-not-found]

        if ctranslate2.get_cuda_device_count():
            return "cuda"
    except ImportError:
        pass
    # Apple Silicon: the Metal backend is far faster than CTranslate2 on CPU.
    return "mlx" if mlx_available() else "cpu"


def _resolve_compute_type(device: str, requested: str) -> str:
    return requested if requested != "auto" else "float16" if device == "cuda" else "float32"


def _sha256(path: Path, cache_dir: Path | None = None) -> str:
    return file_digest(path, cache_dir)


def _optional_float(value: object) -> float | None:
    return float(str(value)) if value is not None else None
