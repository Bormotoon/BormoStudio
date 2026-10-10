from pathlib import Path

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement
from studio.stages.camera_binding import evidence_windows, suggest_bindings
from studio.stages.camera_selection import WIDE_SHOT


def _items(speaking: dict[str, float], faces: int = 1) -> list[dict]:
    return [{"speaker": speaker, "speaking": value, "faces": faces}
            for speaker, value in speaking.items() for _ in range(2)]


def test_close_ups_follow_speaking_contrast_and_wide_needs_people() -> None:
    evidence = {
        "cam-a": _items({"Аня": 2.0, "Борис": -1.0}),
        "cam-b": _items({"Аня": -0.5, "Борис": 1.5}),
        "cam-w": _items({"Аня": 1.0, "Борис": 1.0}, faces=2),
    }
    result = suggest_bindings(evidence, ["Аня", "Борис"])
    assert result["speaker_cameras"] == {"Аня": "cam-a", "Борис": "cam-b", WIDE_SHOT: "cam-w"}
    assert result["unresolved"] == []


def test_weak_or_missing_evidence_is_left_unresolved() -> None:
    evidence = {"cam-a": _items({"Аня": 0.2, "Борис": 0.0}),
                "cam-b": [{"speaker": "Аня", "speaking": None, "faces": 1}]}
    result = suggest_bindings(evidence, ["Аня", "Борис"])
    assert result["speaker_cameras"] == {}
    assert result["unresolved"] == ["Аня", "Борис"]


def test_windows_use_longest_covered_turns_in_file_time() -> None:
    project = Project(Path("/m"), Path("/w"))
    project.assets = [Asset("cam", Path("c.mp4"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 10, 0, 100)]
    turns = [{"start": 0, "end": 30, "speaker": "A"},  # partly before the camera starts
             {"start": 30, "end": 31, "speaker": "A"},  # too short
             {"start": 40, "end": 60, "speaker": "B"},
             {"start": 60, "end": 62, "speaker": "overlap"}]
    windows = evidence_windows(project, turns, "cam", per_speaker=2, length_s=4)
    assert [(w.speaker, w.start_s, w.end_s) for w in windows] == [
        ("A", 8.0, 12.0), ("B", 38.0, 42.0)]


def test_cli_applies_suggestions_without_replacing_choices(tmp_path: Path, monkeypatch,
                                                           capsys) -> None:  # type: ignore[no-untyped-def]
    import json

    from studio.cli.main import main

    work = tmp_path / "_studio"
    work.mkdir()
    project = Project(tmp_path, work)
    turns = work / "turns.json"
    turns.write_text(json.dumps([{"start": 0, "end": 5, "speaker": "A"},
                                 {"start": 5, "end": 9, "speaker": "B"}]))
    project.outputs["speakers"] = [turns]
    project.speaker_cameras = {"A": "manual-cam"}
    project.save(work / "project.json")
    monkeypatch.setattr("studio.stages.camera_binding.collect_evidence",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr("studio.stages.camera_binding.suggest_bindings", lambda evidence, names: {
        "speaker_cameras": {"A": "cam-a", "B": "cam-b"}, "unresolved": []})
    assert main(["cameras", str(tmp_path), "--suggest", "--apply"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["applied"] == {"B": "cam-b"}
    assert Project.load(work / "project.json").speaker_cameras == {"A": "manual-cam", "B": "cam-b"}
