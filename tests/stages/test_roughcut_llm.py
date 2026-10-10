from pathlib import Path

import pytest

from studio.core.settings import Settings, SettingsError, validate_settings
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import stage_uses_llm
from studio.stages.retake_models import RetakeSettings
from studio.stages.roughcut import RoughcutStage, build_edit_list
from studio.stages.roughcut_llm import LlmOptions, locate_quote, run_detectors


def _segment(text: str, start: float) -> Segment:
    words = tuple(
        Word(token, start + index * 0.5, start + index * 0.5 + 0.4)
        for index, token in enumerate(text.split())
    )
    return Segment(words[0].start, words[-1].end, words)


def _transcript() -> Transcript:
    return Transcript(Path("a.wav"), "ru", 30, [
        _segment("раз два три проверка", 0.0),
        _segment("Привет, сегодня говорим о звуке", 3.0),
        _segment("это был вторник, то есть среда", 7.0),
        _segment("как бы важная тема", 11.0),
    ])


def _options(**changes) -> LlmOptions:
    return LlmOptions(**{"detectors": ("start", "slips", "fillers"), **changes})


def test_quote_must_match_words_verbatim() -> None:
    words = _transcript().segments[2].words
    assert locate_quote(words, "был вторник") == (1, 3)
    assert locate_quote(words, "Был, ВТОРНИК!") == (1, 3)
    assert locate_quote(words, "был понедельник") is None
    assert locate_quote(words, "") is None


def test_detectors_snap_to_words_and_reject_unproven_findings() -> None:
    transcript = _transcript()
    prompts = []

    def request(prompt):
        prompts.append(prompt)
        return {
            "start": {"phrase": 1, "reason": "sound check", "confidence": 0.9},
            "slips": [
                {"phrase": 2, "quote": "был вторник,", "reason": "corrected", "confidence": 0.8},
                {"phrase": 2, "quote": "был четверг", "reason": "invented", "confidence": 0.9},
                {"phrase": 99, "quote": "это", "reason": "bad id", "confidence": 0.9},
                {"phrase": 2, "quote": "это", "reason": "weak", "confidence": 0.2},
                {"phrase": 2, "quote": "это", "reason": "bad", "confidence": "high"},
            ],
            "fillers": [{"phrase": 3, "quote": "как бы", "reason": "empty", "confidence": 0.7}],
        }

    result = run_detectors(transcript, [], request, _options())
    assert "[1] 3.0s: Привет, сегодня говорим о звуке" in prompts[0]
    assert [(item.detector, item.start, item.end) for item in result.findings] == [
        ("start", 0.0, 3.0), ("slips", 7.5, 8.4), ("fillers", 11.0, 11.9),
    ]
    reasons = sorted(item["why"] for item in result.rejected)
    assert reasons == [
        "below llm_min_confidence", "invalid confidence",
        "quote not found verbatim in phrase", "unknown phrase",
    ]


def test_start_beyond_limit_and_bad_payloads_are_reported() -> None:
    result = run_detectors(
        _transcript(), [],
        lambda prompt: {"start": {"phrase": 3, "reason": "late"}, "slips": "nope"},
        _options(max_start_s=5.0),
    )
    assert not result.findings
    assert {item["why"] for item in result.rejected} == {
        "start is beyond llm_max_start_s", "not a list",
    }

    def broken(prompt):
        raise ValueError("Could not parse JSON")

    failed = run_detectors(_transcript(), [], broken, _options())
    assert failed.failures == ["chunk 0: Could not parse JSON"]


def test_llm_suggestions_are_unaccepted_and_fillers_stay_markers() -> None:
    transcript = _transcript()
    result = run_detectors(transcript, [], lambda prompt: {
        "start": None,
        "slips": [{"phrase": 2, "quote": "был вторник", "reason": "fix", "confidence": 0.9}],
        "fillers": [{"phrase": 3, "quote": "как бы", "reason": "empty", "confidence": 0.9}],
    }, _options())
    options = dict(mode="cut", minimum_pause_s=5, keep_pause_s=0.4, edge_pad_s=0.2, silence=None)
    edit = build_edit_list(transcript, llm=result, **options)
    llm_cuts = [cut for cut in edit.cuts if cut.reason == "llm"]
    assert len(llm_cuts) == 1 and not llm_cuts[0].accepted
    assert llm_cuts[0].note.startswith("slips: был вторник")
    assert any(start < 7.6 < end for start, end in edit.keep)
    assert [marker.kind for marker in edit.markers] == ["llm-filler"]
    applied = build_edit_list(transcript, llm=result, llm_apply="cut", **options)
    assert any(cut.reason == "llm" and cut.accepted for cut in applied.cuts)
    assert not any(start < 7.6 < end for start, end in applied.keep)


def test_delivery_choice_keeps_an_earlier_take_only_when_applied() -> None:
    line = "мы сегодня говорим про звук"
    words = []
    for start in (0.0, 4.0, 8.0):
        for index, token in enumerate(line.split()):
            words.append(Word(token, start + index * 0.5, start + index * 0.5 + 0.4))
    transcript = Transcript(Path("a.wav"), "ru", 10.4, [Segment(0, 10.4, tuple(words))])
    settings = RetakeSettings(detect_retakes=True, retake_min_words=4, retake_max_gap_s=3.0)
    from studio.stages.retakes import detect_retakes

    groups = detect_retakes(transcript.words, settings)
    assert len(groups) == 1 and len(groups[0].takes) == 3
    prompts = []
    result = run_detectors(transcript, groups, lambda prompt: prompts.append(prompt) or {
        "choices": [
            {"group": 0, "take": 1, "reason": "cleanest", "confidence": 0.8},
            {"group": 3, "take": 0, "reason": "bad", "confidence": 0.8},
        ],
    }, LlmOptions(detectors=("delivery",)))
    assert "take 2:" in prompts[0]
    assert result.delivery == {0: (1, "cleanest", 0.8)}
    options = dict(
        mode="cut", minimum_pause_s=5, keep_pause_s=0.4, edge_pad_s=0.0, silence=None,
        retakes=settings, llm=result,
    )
    suggested = build_edit_list(transcript, **options)
    assert [cut.reason for cut in suggested.cuts] == ["retake"]
    assert suggested.markers[0].kind == "llm-delivery"
    applied = build_edit_list(transcript, llm_apply="cut", **options)
    llm_cuts = [(cut.start, cut.end) for cut in applied.cuts if cut.reason == "llm"]
    assert llm_cuts == [(0.0, 4.0), (8.0, groups[0].takes[-1].end)]
    assert applied.keep[0] == (4.0, 8.0)


def test_settings_and_runner_ownership() -> None:
    settings = Settings()
    validate_settings(settings)
    settings.roughcut.llm_detectors = ["start", "magic"]
    with pytest.raises(SettingsError):
        validate_settings(settings)
    stage = RoughcutStage()
    assert not stage_uses_llm(stage, {"roughcut": {"llm": False}})
    assert stage_uses_llm(stage, {"roughcut": {"llm": True}})
    assert stage.requirements({"roughcut": {"llm": True, "model": "gemma"}})[0].kind == "llama"
