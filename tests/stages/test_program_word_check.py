from pathlib import Path

from studio.core.transcript import Segment, Transcript, Word
from tools.check_program import compare_words


def transcript(tokens: list[str], shift: float = 0) -> Transcript:
    words = tuple(Word(text, index + shift, index + shift + 0.5)
                  for index, text in enumerate(tokens))
    return Transcript(Path("audio.wav"), "en", 10, [Segment(0, 10, words)])


def test_word_check_flags_single_missing_word_and_total_loss() -> None:
    expected = transcript("one two three four five six seven".split())
    assert compare_words(expected, expected)["status"] == "passed"
    assert compare_words(expected, transcript([]))["status"] == "failed"
    missing = compare_words(expected, transcript("one two four five six seven".split()))
    assert missing["status"] == "failed"
    assert missing["differences"][0]["expected"][0]["text"] == "three"


def test_word_check_does_not_accept_empty_reference_or_shifted_speech() -> None:
    expected = transcript("one two three four five six seven".split())
    assert compare_words(transcript([]), transcript([]))["status"] == "inconclusive"
    assert compare_words(expected, transcript(
        "one two three four five six seven".split(), shift=1,
    ))["status"] == "failed"
