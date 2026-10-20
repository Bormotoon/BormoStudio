from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.core.transcript import Segment as TranscriptSegment
from studio.core.transcript import Transcript, Word
from studio.stages.speaker_cascade import (
    LLM_SCHEMA,
    Segment,
    apply_llm_answers,
    audio_votes,
    build_source,
    cached_worker,
    calibrate,
    cascade_sources,
    cluster_labels,
    decide,
    llm_windows,
    posterior,
    run_cascade,
    segment_words,
    vision_step,
)


def _project(tmp_path: Path, assets: list[Asset], length: float = 120) -> Project:
    project = Project(tmp_path, tmp_path / "work")
    project.assets = assets
    project.placements = [SourcePlacement(asset.id, 0, 0, length) for asset in assets]
    return project


def _transcript(tmp_path: Path, length: float = 120, step: float = 2.0) -> Transcript:
    words = tuple(Word(f"w{index}", index * step, index * step + step * 0.8)
                  for index in range(int(length / step)))
    return Transcript(tmp_path / "rec.wav", "ru", length,
                      [TranscriptSegment(0, length, words)],
                      metadata={"source_asset_id": "rec"})


def _turns(first: str, second: str, switch: float = 60, length: float = 120) -> dict:
    return {"turns": [{"start": 0, "end": switch, "speaker": first},
                      {"start": switch, "end": length, "speaker": second}],
            "embeddings": {}}


def test_confidence_grows_with_independent_agreeing_evidence() -> None:
    one = decide({"a": {"P": 1.0}}, {"a": 1.0})
    three = decide({k: {"P": 1.0} for k in "abc"}, {k: 1.0 for k in "abc"})
    split = decide({"a": {"P": 1.0}, "b": {"P": 1.0}, "c": {"Q": 1.0}},
                   {"a": 1.0, "b": 1.0, "c": 1.0})
    assert one == ("P", pytest.approx(0.7))
    assert three[0] == "P" and three[1] > 0.95
    assert split[0] == "P" and split[1] < 0.7
    assert decide({}, {}) == ("unknown", 0.0)


def test_phrases_split_at_pauses_and_length() -> None:
    words = [Word("a", 0, 0.5), Word("b", 0.6, 1.0), Word("c", 2.0, 2.5)]
    assert [(s.start, s.end, s.text) for s in segment_words(words)] == [
        (0, 1.0, "a b"), (2.0, 2.5, "c")]
    long = [Word(str(i), i * 0.5, i * 0.5 + 0.4) for i in range(20)]
    assert all(s.end - s.start <= 6.0 for s in segment_words(long, max_length_s=6.0))


def test_labels_join_by_talking_together_and_by_voice(tmp_path: Path) -> None:
    rec = Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)
    cam = Asset("cam", Path("cam.mp4"), "video", AssetRole.CAMERA, group_id="host")
    late = Asset("late", Path("late.mp4"), "video", AssetRole.CAMERA, group_id="host")
    a = build_source(rec, SourcePlacement("rec", 0, 0, 120), _turns("A", "B")["turns"],
                     {"A": [1, 0], "B": [0, 1]}, 300)
    b = build_source(cam, SourcePlacement("cam", 0, 0, 120), _turns("Y", "X")["turns"],
                     {}, 300)
    c = build_source(late, SourcePlacement("late", 200, 0, 90),
                     [{"start": 0, "end": 40, "speaker": "K"},
                      {"start": 40, "end": 90, "speaker": "L"}],
                     {"K": [0.1, 1], "L": [1, 0.1]}, 300)
    assert a and b and c
    names, _report = cluster_labels([a, b, c])
    assert names[("rec", "A")] == names[("cam", "Y")] == names[("late", "L")]
    assert names[("rec", "B")] == names[("cam", "X")] == names[("late", "K")]
    assert names[("rec", "A")] != names[("rec", "B")]


def test_copies_of_one_recorder_vote_once(tmp_path: Path) -> None:
    front = Asset("front", Path("f.wav"), "audio", AssetRole.RECORDER)
    rear = Asset("rear", Path("r.wav"), "audio", AssetRole.RECORDER)
    sources = [build_source(asset, SourcePlacement(asset.id, 0, 0, 120),
                            _turns("A", "B")["turns"], {}, 120) for asset in (front, rear)]
    segment = Segment(0, 10, 12, "x")
    names = {(asset_id, label): label for asset_id in ("front", "rear") for label in "AB"}
    audio_votes([segment], [s for s in sources if s], names)
    assert list(segment.votes) == ["recorder:"]
    assert segment.votes["recorder:"] == {"A": 1.0}


def test_recorder_copies_are_diarized_once(tmp_path: Path) -> None:
    assets = [Asset(name, Path(f"{name}.wav"), "audio", AssetRole.RECORDER)
              for name in ("rec", "rear", "mix")]
    assets.append(Asset("cam", Path("cam.mp4"), "video", AssetRole.CAMERA))
    project = _project(tmp_path, assets)
    chosen = [asset.id for asset, _p in cascade_sources(project, _transcript(tmp_path), 30)]
    assert chosen[0] == "rec" and "cam" in chosen and len(chosen) == 2


def _three_sources(tmp_path: Path) -> tuple[Project, dict[str, dict]]:
    assets = [Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER),
              Asset("camA", Path("a.mp4"), "video", AssetRole.CAMERA, group_id="ведущий"),
              Asset("camB", Path("b.mp4"), "video", AssetRole.CAMERA, group_id="гость")]
    raw = {"rec": _turns("A", "B"), "camA": _turns("X", "Y"), "camB": _turns("Y", "X")}
    return _project(tmp_path, assets), raw


def test_agreeing_sources_stop_the_cascade_after_audio(tmp_path: Path) -> None:
    project, raw = _three_sources(tmp_path)

    def worker(_request: dict) -> dict:
        raise AssertionError("vision must not run")

    def complete(_prompt: str) -> Any:
        raise AssertionError("the language model must not run")

    with patch("studio.stages.speaker_cascade.diarize_sources", return_value=raw):
        segments, report = run_cascade(project, _transcript(tmp_path), {}, tmp_path,
                                       lambda asset: tmp_path / asset.path,
                                       worker=worker, complete=complete)
    assert all(item.confidence >= 0.95 and item.step == "audio" for item in segments)
    assert {item.speaker for item in segments if item.end <= 60} == {"SPEAKER_00"}
    assert {item.speaker for item in segments if item.start >= 60} == {"SPEAKER_01"}
    assert [step["step"] for step in report["steps"]] == ["audio"]


def test_language_model_decides_only_the_open_phrases(tmp_path: Path) -> None:
    project, raw = _three_sources(tmp_path)
    # The host camera hears the change late (60-80 s) and errs at 90-100 s too;
    # in 60-80 s the guest camera is silent, so the sound alone is split.
    raw["camA"] = {"turns": [{"start": 0, "end": 80, "speaker": "X"},
                             {"start": 80, "end": 90, "speaker": "Y"},
                             {"start": 90, "end": 100, "speaker": "X"},
                             {"start": 100, "end": 120, "speaker": "Y"}], "embeddings": {}}
    raw["camB"] = {"turns": [{"start": 0, "end": 60, "speaker": "Y"},
                             {"start": 80, "end": 120, "speaker": "X"}], "embeddings": {}}
    asked: list[str] = []

    def complete(prompt: str) -> Any:
        # A model that reads the conversation right: the change is at 60 s.
        asked.append(prompt)
        answers = []
        for line in prompt.splitlines():
            if line.startswith("[") and "] ?:" in line:
                number, text = int(line.split("]")[0][1:]), line.split("?:", 1)[1]
                start = int(text.split()[0][1:]) * 2
                answers.append({"line": number, "confidence": 0.9,
                                "speaker": "SPEAKER_00" if start < 60 else "SPEAKER_01"})
        return {"answers": answers + [{"line": 0, "speaker": "SPEAKER_01", "confidence": 1}]}

    with patch("studio.stages.speaker_cascade.diarize_sources", return_value=raw):
        segments, report = run_cascade(project, _transcript(tmp_path),
                                       {"cascade_vision": False}, tmp_path,
                                       lambda asset: tmp_path / asset.path,
                                       complete=complete)
    disputed = [item for item in segments if 62 <= item.start < 80]
    assert disputed and all(item.step == "llm" for item in disputed)
    assert all(item.speaker == "SPEAKER_01" for item in disputed)
    # Control lines it answered right make the model count.
    assert report["steps"][-1]["reliability"]["llm"] > 0.8
    # A line the audio was sure of is never overturned by the model.
    assert segments[0].speaker == "SPEAKER_00"
    assert asked and "?:" in asked[0] and "SPEAKER_00 0." not in asked[0]
    assert LLM_SCHEMA["required"] == ["answers"]


def test_close_up_speaking_votes_for_its_person(tmp_path: Path) -> None:
    cam = Asset("cam", Path("cam.mp4"), "video", AssetRole.CAMERA, group_id="гость")
    project = _project(tmp_path, [cam])
    source = build_source(cam, project.placements[0], _turns("X", "Y")["turns"], {}, 120)
    assert source is not None
    segments = [Segment(0, 10, 11, "a"), Segment(1, 20, 21, "b")]
    requests: list[dict] = []

    def worker(request: dict) -> dict:
        requests.append(request)
        return {"active_speaker": True, "windows": [
            {"faces": 1, "speaking": 3.0}, {"faces": 1, "speaking": -3.0}]}

    votes = vision_step(project, segments, [0, 1], [source], {"camera:гость": "Q"},
                        ["P", "Q"], worker)
    assert votes == 2 and len(requests) == 1
    assert segments[0].votes["vision:camera:гость"]["Q"] > 0.9
    assert segments[1].votes["vision:camera:гость"]["P"] > 0.9


def test_model_answers_are_checked() -> None:
    segments = {1: Segment(1, 0, 1, "a"), 2: Segment(2, 1, 2, "b")}
    applied = apply_llm_answers(segments, {1}, ["P", "Q"], {"answers": [
        {"line": 1, "speaker": "P", "confidence": 0.8},
        {"line": 2, "speaker": "P", "confidence": 0.8},  # not open
        {"line": 1, "speaker": "Z", "confidence": 0.8},  # nobody
        {"line": 1, "speaker": "Q", "confidence": "high"},
    ]})
    assert applied == 1 and segments[1].votes["llm"] == {"P": 1.0}
    assert not segments[2].votes
    with pytest.raises(ValueError):
        apply_llm_answers(segments, {1}, ["P"], ["not", "a", "dict"])


def test_uncertain_neighbours_share_one_request() -> None:
    segments = [Segment(i, i, i + 1, str(i)) for i in range(60)]
    windows = llm_windows(segments, [10, 12, 50], context=3, control=0)
    assert [open_lines for _shown, open_lines in windows] == [[10, 12], [50]]
    assert windows[0][0] == list(range(7, 16))
    controlled = llm_windows(segments, [10, 12, 50], context=3, control=2)
    assert [len(open_lines) for _shown, open_lines in controlled] == [4, 3]
    assert set(controlled[0][1]) - {10, 12} <= set(range(7, 16)) - {10, 12}


def test_frame_grid_follows_the_placement() -> None:
    cam = Asset("cam", Path("c.mp4"), "video", AssetRole.CAMERA)
    source = build_source(cam, SourcePlacement("cam", 100, 10, 50, 1.0),
                          [{"start": 20, "end": 30, "speaker": "X"}], {}, 500)
    assert source is not None
    assert source.seconds(110, 120)[0] == pytest.approx(10.0)
    assert source.seconds(100, 110)[0] == 0
    assert np.asarray(source.active).shape[1] == 500


def test_a_failing_vision_request_costs_only_its_part(tmp_path: Path) -> None:
    cam = Asset("cam", Path("cam.mp4"), "video", AssetRole.CAMERA, group_id="гость")
    project = _project(tmp_path, [cam])
    source = build_source(cam, project.placements[0], _turns("X", "Y")["turns"], {}, 120)
    assert source is not None
    segments = [Segment(i, i * 2.0, i * 2.0 + 1.5, str(i)) for i in range(5)]
    calls = 0

    def worker(request: dict) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("worker crashed")
        return {"active_speaker": True,
                "windows": [{"faces": 1, "speaking": 2.0} for _w in request["windows"]]}

    errors: list[str] = []
    votes = vision_step(project, segments, list(range(5)), [source], {"camera:гость": "Q"},
                        ["P", "Q"], worker, errors, chunk=2)
    assert calls == 3 and votes == 3 and len(errors) == 1
    assert not segments[0].votes and segments[4].votes


def test_voices_compare_around_their_common_component() -> None:
    # Raw cosines of every pair are above 0.99; only the deviations tell people apart.
    base = np.full(8, 10.0)
    p, q = base.copy(), base.copy()
    p[0] += 1
    q[1] += 1
    cams = [Asset(name, Path(f"{name}.mp4"), "video", AssetRole.CAMERA) for name in "abc"]
    sources = []
    for index, asset in enumerate(cams):
        turns = [{"start": 0, "end": 20, "speaker": "S0"},
                 {"start": 20, "end": 40, "speaker": "S1"}]
        vectors = {"S0": (p if index != 1 else q).tolist(), "S1": (q if index != 1 else p).tolist()}
        source = build_source(asset, SourcePlacement(asset.id, index * 100, 0, 40), turns,
                              vectors, 400)
        assert source is not None
        sources.append(source)
    names, _report = cluster_labels(sources)
    assert names[("a", "S0")] == names[("b", "S1")] == names[("c", "S0")]
    assert names[("a", "S0")] != names[("a", "S1")]


def test_an_unreliable_source_is_outvoted_and_one_source_is_not_certain() -> None:
    votes = {"good": {"P": 1.0}, "fair": {"P": 1.0}, "bad": {"Q": 1.0}}
    weights = dict.fromkeys(votes, 1.0)
    speaker, sure = posterior(votes, weights, {"good": 0.95, "fair": 0.9, "bad": 0.6})
    assert speaker == "P" and sure > 0.99
    alone = posterior({"good": {"P": 1.0}}, {"good": 1.0}, {"good": 0.93}, ["P", "Q"])
    assert alone == ("P", pytest.approx(0.93))
    # A vote with little behind it moves toward chance.
    faint = posterior({"good": {"P": 1.0}}, {"good": 0.2}, {"good": 0.93}, ["P", "Q"])
    assert 0.5 < faint[1] < 0.7


def test_reliability_follows_agreement_with_the_others() -> None:
    segments = []
    for index in range(40):
        truth = "P" if index % 2 else "Q"
        wrong = "Q" if truth == "P" else "P"
        segment = Segment(index, index, index + 1, str(index))
        segment.votes = {"a": {truth: 1.0}, "b": {truth: 1.0},
                         "c": {wrong if index % 4 == 0 else truth: 1.0}}
        segment.weights = dict.fromkeys(segment.votes, 1.0)
        segment.decide("audio")
        segments.append(segment)
    reliability = calibrate(segments, ["P", "Q"], "audio", {item.id for item in segments})
    assert reliability["a"] > 0.9 and reliability["b"] > 0.9
    assert 0.6 < reliability["c"] < 0.85
    assert all(item.speaker == ("P" if item.id % 2 else "Q") for item in segments)


def test_light_asd_results_are_reused(tmp_path: Path) -> None:
    video = tmp_path / "cam.mp4"
    video.write_bytes(b"video")
    calls: list[list] = []

    def worker(request: dict) -> dict:
        calls.append(request["windows"])
        return {"active_speaker": True,
                "windows": [{"faces": 1, "speaking": float(low)}
                            for low, _high in request["windows"]]}

    cached = cached_worker(worker, tmp_path / "asd")
    request = {"mode": "camera_evidence", "video": str(video), "windows": [[1, 2], [3, 4]]}
    first = cached(request)
    second = cached({**request, "windows": [[3, 4], [5, 6]]})
    assert calls == [[[1, 2], [3, 4]], [[5, 6]]]
    assert first["windows"][1] == second["windows"][0] == {"faces": 1, "speaking": 3.0}


def test_cascade_results_follow_the_cascade_version(tmp_path: Path) -> None:
    from studio.stages import speaker_cascade
    from studio.stages.speakers import SpeakersStage

    project = _project(tmp_path, [Asset("rec", Path("rec.wav"), "audio", AssetRole.RECORDER)])
    (tmp_path / "rec.wav").write_bytes(b"x")
    timeline = tmp_path / "timeline.json"
    _transcript(tmp_path).save(timeline)
    project.transcripts["timeline"] = timeline
    settings = {"speakers": {"method": "cascade", "cascade_llm": False}}
    before = SpeakersStage().fingerprint(project, settings)
    with patch.object(speaker_cascade, "CASCADE_VERSION", "cascade-test"):
        assert SpeakersStage().fingerprint(project, settings) != before
