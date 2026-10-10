import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import studio.stages.transcribe as transcribe
from studio.core.autotune import whisper_candidates
from studio.core.settings import load_settings
from studio.stages.transcribe import (
    MlxWhisperBackend,
    TranscribeSettings,
    TranscribeStage,
    whisper_backend,
)


@pytest.fixture
def apple_silicon(monkeypatch):  # type: ignore[no-untyped-def]
    calls = []

    def fake_transcribe(audio: str, **kwargs: object) -> dict:
        calls.append((audio, kwargs))
        return {"language": "ru", "segments": [
            {"start": 0.1, "end": 1.2, "text": " Привет мир", "avg_logprob": -0.2,
             "words": [{"word": " Привет", "start": 0.1, "end": 0.6, "probability": 0.9},
                       {"word": " мир", "start": 0.7, "end": 1.2, "probability": 0.8}]}]}

    module = types.ModuleType("mlx_whisper")
    module.transcribe = fake_transcribe  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(transcribe, "mlx_available", lambda: True)
    return calls


def test_auto_device_uses_mlx_on_apple_silicon_without_cuda(apple_silicon) -> None:  # type: ignore[no-untyped-def]
    with patch.dict(sys.modules, {"ctranslate2": None}):
        assert transcribe._resolve_device("auto") == "mlx"
        assert isinstance(whisper_backend(TranscribeSettings()), MlxWhisperBackend)
    assert transcribe._resolve_device("cpu") == "cpu"
    requirement = TranscribeStage().requirements({"transcribe": {"device": "auto"}})[0]
    assert requirement.value == "mlx-whisper"


def test_mlx_backend_maps_words_and_options(apple_silicon, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    audio = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "anullsrc=r=16000:cl=mono", "-t", "2", str(audio)], check=True)
    settings = TranscribeSettings(model="large-v3-turbo", glossary=("Studio",), mode="quality")
    result = MlxWhisperBackend(settings).transcribe(audio, settings)
    assert [(w.text, w.start) for w in result.words] == [("Привет", 0.1), ("мир", 0.7)]
    assert result.device == "mlx" and result.duration == pytest.approx(2.0, abs=0.05)
    _audio, options = apple_silicon[0]
    assert options["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"
    assert options["word_timestamps"] is True and options["condition_on_previous_text"]
    assert "Studio" in str(options["initial_prompt"])


def test_mlx_is_a_valid_device_and_tunes_with_one_trial() -> None:
    assert load_settings([], ["transcribe.device=mlx"]).transcribe.device == "mlx"
    assert whisper_candidates({"available": True, "mlx": True}) == ("mlx", ["float16"], [1])


def test_mlx_needs_apple_silicon_and_the_package() -> None:
    import importlib.util
    import platform

    found = object()
    with patch.object(sys, "platform", "darwin"), patch.object(
        platform, "machine", lambda: "arm64"
    ), patch.object(importlib.util, "find_spec", lambda name: found):
        assert transcribe.mlx_available() is True
        with patch.object(importlib.util, "find_spec", lambda name: None):
            assert transcribe.mlx_available() is False
        with patch.object(platform, "machine", lambda: "x86_64"):
            assert transcribe.mlx_available() is False
    with patch.object(sys, "platform", "linux"):
        assert transcribe.mlx_available() is False
