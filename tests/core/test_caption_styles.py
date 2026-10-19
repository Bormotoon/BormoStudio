import pytest

from studio.core.caption_styles import (
    CaptionOptions,
    TimedWord,
    build_cues,
    display_word,
    load_presets,
    preset_names,
    to_styled_ass,
    wrap,
)
from studio.core.settings import SettingsError, load_settings


def _words(text: str, step: float = 0.3) -> list[TimedWord]:
    return [TimedWord(word, i * step, i * step + step - 0.02)
            for i, word in enumerate(text.split())]


def test_presets_are_shared_with_the_web_editor() -> None:
    assert {"forge", "hormozi", "karaoke", "one_word"} <= set(preset_names())
    assert load_presets()["presets"]["hormozi"]["render"]["highlight"] == "word"


def test_preset_render_defaults_and_explicit_overrides() -> None:
    hormozi = CaptionOptions.from_settings({"caption_preset": "hormozi"})
    assert (hormozi.highlight, hormozi.text_case, hormozi.max_words) == ("word", "upper", 4)
    plain = CaptionOptions.from_settings({"caption_preset": "hormozi", "caption_highlight":
                                          "none", "caption_max_words": 0})
    assert (plain.highlight, plain.max_words) == ("none", 0)
    with pytest.raises(ValueError):
        CaptionOptions.from_settings({"caption_preset": "nope"})


def test_cues_split_at_pauses_word_limits_and_keep_gaps() -> None:
    options = CaptionOptions.from_settings({"caption_preset": "hormozi"})
    words = _words("one two three four five six") + [TimedWord("after", 5.0, 5.4)]
    cues = build_cues(words, options)
    assert [cue.text for cue in cues] == ["ONE TWO THREE FOUR", "FIVE SIX", "AFTER"]
    pairs = zip(cues, cues[1:], strict=False)
    assert all(b.start - a.end >= options.min_gap_s - 1e-9 for a, b in pairs)
    one = CaptionOptions.from_settings({"caption_preset": "one_word"})
    assert [cue.text for cue in build_cues(_words("Hi, there."), one)] == ["HI", "THERE"]


def test_punctuation_whisper_split_off_stays_with_its_word() -> None:
    # One word per cue makes a split between "«Почему" and "?»" visible.
    one = CaptionOptions(max_words=1)
    words = [TimedWord("«Почему", 0.0, 0.7, "A"), TimedWord("?»", 0.9, 1.0, "B"),
             TimedWord("меня", 1.1, 1.3, "B")]
    assert [cue.text for cue in build_cues(words, one)] == ["«Почему?»", "меня"]


def test_hyphenated_words_whisper_split_stay_whole() -> None:
    one = CaptionOptions(max_words=1, text_case="upper")
    words = [TimedWord("кто", 0.0, 0.3, "A"), TimedWord("-то", 0.3, 0.5, "A"),
             TimedWord("конкуренты", 0.6, 1.2, "A"), TimedWord("—", 1.3, 1.4, "A"),
             TimedWord("да", 1.5, 1.7, "A")]
    assert [cue.text for cue in build_cues(words, one)] == [
        "КТО-ТО", "КОНКУРЕНТЫ", "—", "ДА"]


def test_text_transforms_and_censoring() -> None:
    options = CaptionOptions(text_case="upper", punctuation="periods",
                             censor=("бля*",), censor_style="middle")
    assert display_word("бляха,", options) == "Б***А"
    assert display_word("ok.", options) == "OK"
    assert display_word("well...", CaptionOptions(punctuation="all")) == "well"
    assert display_word("bad", CaptionOptions(censor=("bad",), censor_style="whole")) == "***"


def test_lines_are_bottom_heavy() -> None:
    assert wrap(["a", "bb", "cccc", "dddd"], 8, 2) == {2}
    assert wrap(["short"], 8, 2) == set()
    assert wrap(["a", "b", "c"], 2, 1) == set()


@pytest.mark.parametrize("mode", ["none", "word", "fill", "reveal", "pop", "karaoke"])
def test_highlight_modes_write_valid_events(mode: str) -> None:
    options = CaptionOptions.from_settings({"caption_preset": "forge",
                                            "caption_highlight": mode})
    script = to_styled_ass(build_cues(_words("we ship today"), options), options)
    events = [line for line in script.splitlines() if line.startswith("Dialogue:")]
    assert events and "Style: Default,BigNoodleTooOblique" in script
    if mode == "karaoke":
        assert len(events) == 1 and "\\kf" in events[0]
    elif mode != "none":
        assert len(events) == 3  # one per spoken word
    if mode == "pop":
        assert "\\t(0," in script


def test_landscape_reels_scale_the_look() -> None:
    portrait = CaptionOptions.from_settings({"caption_preset": "forge"})
    landscape = CaptionOptions.from_settings({"caption_preset": "forge"})
    landscape = type(landscape)(**{**{f: getattr(landscape, f) for f in landscape.__slots__},
                                   "play_res": (1920, 1080)})
    assert "PlayResX: 1920" in to_styled_ass([], landscape)
    assert landscape.scale == pytest.approx(portrait.scale * 1080 / 1920)


@pytest.mark.parametrize("override", [
    "reels.caption_preset=nope", "reels.caption_highlight=blink", "reels.caption_case=camel",
    "reels.caption_max_words=-2", "reels.caption_highlight_color=red",
    "reels.caption_censor_style=all", "reels.caption_width_ratio=2",
])
def test_invalid_caption_settings(override: str) -> None:
    with pytest.raises(SettingsError):
        load_settings([], [override])
