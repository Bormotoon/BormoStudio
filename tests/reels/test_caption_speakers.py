import json
from pathlib import Path

from studio.core.caption_styles import CaptionOptions, TimedWord, build_cues, cue_events
from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.reels.analysis.speaker_context import speaker_at, speaker_turns


def _words() -> list[TimedWord]:
    return [TimedWord("one", 0.0, 0.3, "A"), TimedWord("two", 0.35, 0.6, "A"),
            TimedWord("three", 0.65, 0.9, "B"), TimedWord("four", 0.95, 1.2, "A")]


def test_a_change_of_speaker_starts_a_cue_and_takes_its_colour() -> None:
    options = CaptionOptions.from_settings({"caption_speaker_colors": ["#FF0000", "#00FF00"]})
    cues = build_cues(_words(), options)
    assert [[w.text for w in cue.words] for cue in cues] == [["one", "two"], ["three"], ["four"]]
    events = cue_events(cues, options)
    assert "\\1c&H0000FF&" in events[0] and "\\1c&H00FF00&" in events[1]
    assert "\\1c&H0000FF&" in events[-1]
    joined = build_cues(_words(), CaptionOptions.from_settings(
        {"caption_split_on_speaker": False}))
    assert len(joined) == 1


def test_vertical_align_and_offset() -> None:
    top = cue_events(build_cues(_words(), CaptionOptions.from_settings(
        {"caption_vertical_align": "top", "caption_split_on_speaker": False})),
        CaptionOptions.from_settings({"caption_vertical_align": "top"}))
    assert "{\\an8" in top[0] and ",Default,,0,0,250,," in top[0]
    centre = CaptionOptions.from_settings(
        {"caption_vertical_align": "center", "caption_vertical_offset": 0.1})
    assert "\\an5\\pos(540,768)" in cue_events(build_cues(_words(), centre), centre)[0]
    plain = CaptionOptions.from_settings({})
    assert ",Default,,0,0,0,," in cue_events(build_cues(_words(), plain), plain)[0]


def test_speaker_turns_follow_the_program_mapping(tmp_path: Path) -> None:
    turns = tmp_path / "diarization.json"
    turns.write_text(json.dumps([{"start": 0, "end": 10, "speaker": "A"},
                                 {"start": 10, "end": 20, "speaker": "B"}]))
    report = tmp_path / "report.json"
    report.write_text('{"time_domain": "timeline"}')
    edited = Transcript(tmp_path / "p.mp4", "en", 8,
                        [Segment(0, 8, (Word("x", 1, 2), Word("y", 6, 7)))],
                        time_domain=TimeDomain.EDITED)
    from studio.core.project import stable_fingerprint

    program = tmp_path / "program.json"
    program.write_text(json.dumps({
        "edited_timing_fingerprint": stable_fingerprint(8, [(1, 2), (6, 7)]),
        "timeline_to_rendered": [
            {"timeline_start": 6, "timeline_end": 10, "edited_start": 0, "edited_end": 4},
            {"timeline_start": 12, "timeline_end": 16, "edited_start": 4, "edited_end": 8}],
    }))
    project = Project(tmp_path, tmp_path / "work", outputs={
        "speakers": [turns, report], "program_report": [program]})
    mapped = speaker_turns(project, edited)
    assert mapped == [(0.0, 4.0, "A"), (4.0, 8.0, "B")]
    assert speaker_at(mapped, 1.5) == "A" and speaker_at(mapped, 6.5) == "B"
    timeline = Transcript(tmp_path / "a.wav", "en", 20, [], time_domain=TimeDomain.TIMELINE)
    assert speaker_turns(project, timeline)[1] == (10.0, 20.0, "B")
