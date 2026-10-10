from pathlib import Path

import pytest

from studio.core.transcript import Segment, Transcript, Word
from studio.core.transcript_sanity import repair_detached_words, speech_clips


def _energy(speech: list[tuple[float, float]]):
    def energy(start: float, end: float) -> float:
        overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in speech)
        return -20.0 if overlap >= 0.5 * (end - start) else -70.0
    return energy


def _transcript(*segments: tuple[str, ...]) -> Transcript:
    built = []
    for words in segments:
        parsed = tuple(Word(text, float(start), float(start) + 0.3)
                       for text, start in (item.rsplit("@", 1) for item in words))
        built.append(Segment(parsed[0].start, parsed[-1].end, parsed))
    return Transcript(Path("camera.wav"), "ru", 600.0, built)


def test_clip_builder_never_spans_long_silence() -> None:
    clips = speech_clips([(14.77, 15.47), (141.0, 150.0), (150.8, 170.0)])
    assert clips == [(14.77, 15.47), (141.0, 170.0)]
    assert speech_clips([(1.0, 1.1), (5.0, 9.0)]) == [(5.0, 9.0)]  # lone blip dropped
    long = speech_clips([(0.0, 20.0), (20.5, 40.0)])
    assert all(end - start <= 30.0 for start, end in long)


def test_glued_first_word_moves_back_to_its_phrase() -> None:
    glued = _transcript(("По@14.8", "новой,@141.7", "давайте@142.1"))
    repaired = repair_detached_words(glued, _energy([(14.7, 15.3), (140.9, 145.0)]))
    words = repaired.words
    assert words[0].text == "По"
    assert words[0].end == pytest.approx(141.68, abs=0.01)
    assert len(repaired.segments) == 1
    assert repaired.metadata["moved_words"][0]["from"] == pytest.approx(14.8)


def test_recorder_case_kak_dumaete() -> None:
    glued = _transcript(("Как@455.5", "думаете,@509.9", "это@510.3"))
    repaired = repair_detached_words(glued, _energy([(455.4, 456.0), (508.9, 512.0)]))
    assert repaired.words[0].start == pytest.approx(509.58, abs=0.01)


def test_real_pause_between_sentences_is_kept() -> None:
    transcript = _transcript(("Да.@10.0", "Следующий@18.5", "вопрос@18.9"))
    repaired = repair_detached_words(transcript, _energy([(9.9, 10.4), (18.4, 19.5)]))
    assert repaired.words[0].start == 10.0
    assert "suspect_words" not in repaired.metadata
    assert len(repaired.segments) == 2


def test_word_in_silence_without_speech_near_the_phrase_is_suspect() -> None:
    transcript = _transcript(("и@30.0", "потом@40.0", "дальше@40.4"))
    repaired = repair_detached_words(transcript, _energy([(39.9, 41.0)]))
    first = repaired.words[0]
    assert first.start == 30.0 and first.probability <= 0.1
    assert repaired.metadata["suspect_words"][0]["text"] == "и"
