"""Shared Whisper engine and project transcription stage."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from platformdirs import user_cache_path

from studio.core.project import Project, stable_fingerprint
from studio.core.transcript import Segment, Transcript, Word, to_srt
from studio.stages.base import Decision, GpuUse, Requirement, StageContext, StageOutput

CACHE_SCHEMA = "studio/transcript-cache/1"
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
    keep_fillers: bool = True
    use_cache: bool = True

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
            keep_fillers=bool(values.get("keep_fillers", True)),
            use_cache=bool(values.get("use_cache", True)),
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
        kwargs: dict[str, Any] = {
            "language": settings.language,
            "word_timestamps": True,
            "vad_filter": True,
            "temperature": TEMPERATURES,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "initial_prompt": settings.prompt() or None,
        }
        if settings.mode == "quality":
            kwargs.update(
                beam_size=settings.quality_beam_size,
                condition_on_previous_text=True,
            )
            generated, info = model.transcribe(str(audio), **kwargs)
        else:
            from faster_whisper import BatchedInferencePipeline  # type: ignore[import-not-found]

            kwargs.update(beam_size=settings.beam_size, condition_on_previous_text=False)
            generated, info = BatchedInferencePipeline(model=model).transcribe(
                str(audio), batch_size=max(1, settings.batch_size), **kwargs
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

    def unload(self) -> None:
        self._model = None
        gc.collect()
        with contextlib.suppress(ImportError, RuntimeError):
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class WhisperEngine:
    def __init__(
        self,
        settings: TranscribeSettings,
        backend: WhisperBackend | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend or FasterWhisperBackend(settings)
        self.cache_dir = cache_dir or user_cache_path("studio") / "transcripts"

    def transcribe(self, path: Path, stream_index: int | None = None) -> Transcript:
        cache_path = self.cache_dir / f"{self._key(path, stream_index)}.json"
        if self.settings.use_cache:
            cached = self._load(cache_path)
            if cached is not None:
                cached.source_audio = path
                return cached
        if stream_index is None:
            transcript = self.backend.transcribe(path, self.settings)
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
                transcript = self.backend.transcribe(decoded, self.settings)
        transcript.source_audio = path
        if self.settings.use_cache:
            self._save(cache_path, transcript)
        return transcript

    def _key(self, path: Path, stream_index: int | None) -> str:
        identity = _sha256(path)
        return stable_fingerprint(
            CACHE_SCHEMA,
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
    gpu = GpuUse.WHISPER

    def requirements(self, settings: dict[str, object]) -> list[Requirement]:
        return [Requirement("package", "faster-whisper", "local speech recognition")]

    def decide(self, project: Project, settings: dict[str, object]) -> Decision:
        requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
        if not requirements.is_file():
            return Decision.blocked("prepare requirements are missing", "run prepare")
        data = json.loads(requirements.read_text(encoding="utf-8"))
        return Decision.run() if data.get("transcripts") else Decision.skip(
            "no source requires transcription"
        )

    def fingerprint(self, project: Project, settings: dict[str, object]) -> str:
        requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
        content = requirements.read_text(encoding="utf-8") if requirements.is_file() else ""
        requested = json.loads(content).get("transcripts", []) if content else []
        assets = {asset.id: asset for asset in project.assets}
        identities = {}
        for asset_id in requested:
            asset = assets.get(asset_id)
            source = project.source_dir / asset.path if asset else None
            identities[asset_id] = (
                {"path": str(source), "sha256": _sha256(source)}
                if source is not None and source.is_file() else {"missing": True}
            )
        return stable_fingerprint(
            "transcribe-v2", content, identities, settings.get("transcribe", {}),
        )

    def run(self, context: StageContext) -> StageOutput:
        requirements_path = context.project.work_dir / "stages" / "prepare" / "requirements.json"
        requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
        asset_by_id = {asset.id: asset for asset in context.project.assets}
        settings = TranscribeSettings.from_mapping(context.settings.get("transcribe", {}))
        engine = WhisperEngine(settings)
        artifacts: list[Path] = []
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
        return StageOutput(tuple(artifacts), {"transcripts": transcripts})


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2  # type: ignore[import-not-found]

        return "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_compute_type(device: str, requested: str) -> str:
    return requested if requested != "auto" else "float16" if device == "cuda" else "float32"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_float(value: object) -> float | None:
    return float(str(value)) if value is not None else None
