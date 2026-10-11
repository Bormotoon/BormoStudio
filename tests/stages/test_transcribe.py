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
    with patch.object(backend, "_ensure_model", return_value=model), patch.object(
        backend, "_speech_clips", return_value=([0.0] * 16000, [(0.0, 1.0)])
    ):
        transcript = backend.transcribe(Path("audio.wav"), settings)
    assert model.transcribe.call_args.kwargs["hotwords"] == "UVF"
    assert model.transcribe.call_args.kwargs["clip_timestamps"] == [0.0, 1.0]
    assert model.transcribe.call_args.kwargs["vad_filter"] is False
    assert model.transcribe.call_args.kwargs["repetition_penalty"] == 1.1
    assert model.transcribe.call_args.kwargs["no_repeat_ngram_size"] == 3
    assert transcript.segments[0].text == "Unaligned speech."
    assert transcript.to_dict()["sentences"][0]["text"] == "Unaligned speech."


def test_quality_mode_feeds_the_model_bounded_blocks() -> None:
    # sony_run: one sequential call over 3.75 h built the features of the whole
    # recording at once and the machine ran out of memory.
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    settings = TranscribeSettings(device="cpu", mode="quality")
    backend = FasterWhisperBackend(settings)
    calls = []

    def transcribe(audio, clip_timestamps, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((len(audio) / 16000, list(clip_timestamps), kwargs.get("initial_prompt")))
        first = clip_timestamps[0]
        word = SimpleNamespace(word="слово", start=first + 0.1, end=first + 0.4, probability=0.9)
        return [SimpleNamespace(start=first, end=first + 0.5, text=f"блок {len(calls)}",
                                words=[word], avg_logprob=-0.1)], SimpleNamespace(language="ru")

    model = Mock()
    model.transcribe.side_effect = transcribe
    hour = 3600 * 16000
    clips = [(start, start + 20.0) for start in range(100, 3500, 60)]
    with patch.object(backend, "_ensure_model", return_value=model), patch.object(
        backend, "_speech_clips", return_value=(np.zeros(hour, np.float32), clips)
    ):
        transcript = backend.transcribe(Path("long.wav"), settings)
    assert len(calls) > 1 and all(length <= 600.1 for length, _clips, _prompt in calls)
    assert all(values[0] == 0.0 for _length, values, _prompt in calls)  # block-relative
    assert calls[1][2] and "блок 1" in calls[1][2]  # the previous block continues as prompt
    starts = [segment.start for segment in transcript.segments]
    assert starts[0] == 100.0 and starts == sorted(starts)
    second_block = clips[len(calls[0][1]) // 2]  # first clip of the second call
    assert abs(transcript.words[1].start - (second_block[0] + 0.1)) < 1e-6
    assert transcript.duration == 3600.0 and transcript.language == "ru"


def test_camera_transcripts_use_the_mode_prepare_asked_for(tmp_path: Path) -> None:
    import json

    from studio.core.project import Asset, AssetKind, AssetRole, Project
    from studio.stages.base import StageContext
    from studio.stages.transcribe import TranscribeStage

    class Recording(_Backend):
        def __init__(self) -> None:
            super().__init__()
            self.modes: dict[str, str] = {}

        def transcribe(self, audio: Path, settings: TranscribeSettings) -> Transcript:
            self.modes[audio.stem] = settings.mode
            return super().transcribe(audio, settings)

    for name in ("cam.wav", "rec.wav"):
        (tmp_path / name).write_bytes(name.encode())
    project = Project(tmp_path, tmp_path / "_studio", assets=[
        Asset("cam", Path("cam.wav"), AssetKind.AUDIO, AssetRole.CAMERA),
        Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER),
    ])
    requirements = project.work_dir / "stages" / "prepare" / "requirements.json"
    requirements.parent.mkdir(parents=True)
    requirements.write_text(json.dumps({"transcripts": ["rec", "cam"],
                                        "transcribe_modes": {"cam": "fast"},
                                        "match_transcripts": {"rec": "fast"}}))
    backend = Recording()
    with patch("studio.stages.transcribe.whisper_backend", return_value=backend), \
            patch("studio.stages.transcribe.user_cache_path", return_value=tmp_path / "cache"), \
            patch("studio.stages.transcribe.energy_probe", return_value=lambda a, b: 0.0):
        output = TranscribeStage().run(StageContext(
            project, {"transcribe": {"mode": "quality"}}, project.work_dir))
    # The recorder: quality for the content, then fast for matching camera words.
    assert backend.modes == {"rec": "fast", "cam": "fast"}
    assert backend.calls == 3
    transcripts = output.project_changes["transcripts"]
    assert transcripts["match:rec"].name == "rec.match.json"
    assert transcripts["rec"].name == "rec.json"
