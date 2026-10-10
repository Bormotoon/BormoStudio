from pathlib import Path
from unittest.mock import patch

from studio.core.transcript import Segment, Transcript, Word
from studio.stages.transcribe import FasterWhisperBackend, TranscribeSettings, WhisperEngine


class _Backend:
    device = "cpu"
    compute_type = "float32"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript:
        self.calls += 1
        return Transcript(audio, "en", 1.0, [Segment(0, 1, (Word("hello", 0, 1),))])


def test_content_cache_reuses_same_source_across_path_mtime_changes(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"same audio")
    second.write_bytes(b"same audio")
    backend = _Backend()
    engine = WhisperEngine(TranscribeSettings(), backend, tmp_path / "cache")
    assert engine.transcribe(first).words[0].text == "hello"
    assert engine.transcribe(second).words[0].text == "hello"
    assert engine.transcribe(second).source_audio == second
    assert backend.calls == 1


def test_cache_changes_with_stream_or_decoding_settings(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    backend = _Backend()
    with patch("studio.stages.transcribe.subprocess.run") as decode:
        WhisperEngine(TranscribeSettings(), backend, tmp_path / "cache").transcribe(source, 0)
        WhisperEngine(TranscribeSettings(), backend, tmp_path / "cache").transcribe(source, 1)
        changed = TranscribeSettings(initial_prompt="context")
        WhisperEngine(changed, backend, tmp_path / "cache").transcribe(source, 1)
        assert [call.args[0][call.args[0].index("-map") + 1]
                for call in decode.call_args_list] == ["0:0", "0:1", "0:1"]
    assert backend.calls == 3


def test_prompt_combines_context_glossary_and_fillers() -> None:
    prompt = TranscribeSettings(
        initial_prompt="Podcast about Studio.", glossary=("Bormotoon",), keep_fillers=True
    ).prompt()
    assert "Bormotoon" in prompt
    assert "ээ" in prompt


def test_oom_retries_lazy_generation_with_smaller_batches() -> None:
    backend = FasterWhisperBackend(TranscribeSettings(device="cpu"))
    attempts = []

    def recognize(audio, settings):
        attempts.append((settings.mode, settings.batch_size))
        if settings.batch_size > 2:
            raise RuntimeError("CUDA out of memory")
        return Transcript(audio, "en", 0.0, [])

    with patch.object(backend, "_transcribe_once", side_effect=recognize), patch.object(
        backend, "unload"
    ) as unload:
        backend.transcribe(Path("audio.wav"), TranscribeSettings(batch_size=8))
    assert attempts == [("fast", 8), ("fast", 4), ("fast", 2)]
    assert unload.call_count == 2


def test_quality_backend_passes_glossary_controls_and_preserves_unaligned_text() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    settings = TranscribeSettings(device="cpu", mode="quality", glossary=("UVF",))
    backend = FasterWhisperBackend(settings)
    model = Mock()
    model.transcribe.return_value = ([SimpleNamespace(
        start=0, end=1, text="Unaligned speech.", words=[], avg_logprob=-0.2,
    )], SimpleNamespace(language="en", duration=1))
    with patch.object(backend, "_ensure_model", return_value=model):
        transcript = backend.transcribe(Path("audio.wav"), settings)
    assert model.transcribe.call_args.kwargs["hotwords"] == "UVF"
    assert model.transcribe.call_args.kwargs["repetition_penalty"] == 1.1
    assert model.transcribe.call_args.kwargs["no_repeat_ngram_size"] == 3
    assert transcript.segments[0].text == "Unaligned speech."
    assert transcript.to_dict()["sentences"][0]["text"] == "Unaligned speech."
