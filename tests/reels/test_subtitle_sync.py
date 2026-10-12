import json
import random
import subprocess
from pathlib import Path

import pytest

import studio.stages.reel_render as reel_render
from studio.core.project import Project
from studio.core.timeline import TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.reels import subtitle_sync
from studio.stages.base import StageContext
from studio.stages.reel_render import ReelRenderStage


def _words(seed: int, drift: float) -> tuple[list, list]:
    rng = random.Random(seed)
    vocabulary = "alpha beta gamma delta echo foxtrot golf hotel india juliet".split()
    reference, heard, at = [], [], 0.0
    for index in range(60):
        text = rng.choice(vocabulary)
        reference.append(subtitle_sync.TimedWord(round(at, 3), round(at + 0.3, 3), text))
        if index % 7 != 3:  # some words go unheard
            spelled = text if index % 11 else text[:-1] + "x"
            heard.append(subtitle_sync.TimedWord(round(at + drift, 3),
                                                 round(at + 0.3 + drift, 3), spelled))
        at += rng.choice((0.35, 0.4, 0.9))
    return reference, heard


@pytest.mark.parametrize(("seed", "drift"), [(1, 0.05), (2, 0.4), (3, -0.3)])
def test_caption_check_matches_forge(forge_snapshot: None, seed: int, drift: float) -> None:
    from podcast_reels_forge.utils import subtitle_sync as forge

    reference, heard = _words(seed, drift)
    ours, report = subtitle_sync.judge(reference, heard, min_match_ratio=0.5,
                                       apply_threshold_s=0.2)
    theirs_ref = [forge.TimedWord(w.start, w.end, w.text) for w in reference]
    theirs_heard = [forge.TimedWord(w.start, w.end, w.text) for w in heard]
    pairs = forge.match_words(theirs_ref, theirs_heard)
    assert subtitle_sync.match_words(reference, heard) == pairs
    expected = (forge.retime(theirs_ref, theirs_heard, pairs) if report.applied
                else theirs_ref)
    assert [(w.start, w.end, w.text) for w in ours] == [
        (w.start, w.end, w.text) for w in expected]
    assert report.verdict == ("in_sync" if abs(drift) < 0.2 else "retimed")


def test_reel_captions_take_the_second_pass(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "program.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=red:s=160x90:r=25:d=6", "-f", "lavfi", "-i", "sine=duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source)],
        check=True)
    words = tuple(Word(text, 1 + i * 0.5, 1.3 + i * 0.5)
                  for i, text in enumerate("one two three four.".split()))
    transcript_path = tmp_path / "edited.json"
    Transcript(source, "en", 6, [Segment(1, 2.8, words)],
               time_domain=TimeDomain.EDITED).save(transcript_path)
    moments = tmp_path / "moments.json"
    moments.write_text(json.dumps([{"start": 1, "end": 2.8, "candidate_id": "a", "score": 9}]))
    project = Project(tmp_path, tmp_path / "work", transcripts={"edited": transcript_path},
                      outputs={"program": [source], "reels": [moments]})
    heard = [subtitle_sync.TimedWord(0.6 + 0.3 + i * 0.5, 0.6 + 0.6 + i * 0.5, text)
             for i, text in enumerate("one two three four".split())]
    monkeypatch.setattr(reel_render, "load_sync_model", lambda _settings: object())
    monkeypatch.setattr(reel_render, "heard_words", lambda *_args: heard)
    settings = {"reels": {"render": True, "covers": False, "loudness_lufs": 0,
                          "min_score": 0, "min_duration_s": 0, "caption_whisper_sync": True,
                          "preview_reel": False}}
    ReelRenderStage().run(StageContext(project, settings, project.work_dir))
    folder = project.work_dir / "stages" / "reel_render"
    report = json.loads((folder / "subtitle_sync.json").read_text())
    assert report["clips"][0]["verdict"] == "retimed"
    # The reel starts 0.6 s before the first word; the heard words are 0.3 s later.
    assert "00:00:00,900 -->" in (folder / "reel-001.srt").read_text()
    # The reel transcript the editor opens keeps the original timings.
    assert Transcript.load(folder / "reel-001.json").words[0].start == pytest.approx(0.6)
