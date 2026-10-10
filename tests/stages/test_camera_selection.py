from pathlib import Path

import pytest

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import EditMap, KeepRange, SourcePlacement
from studio.stages.camera_selection import WIDE_SHOT, CameraPolicy, speaker_camera_edit
from studio.stages.program import program_pieces


def project() -> Project:
    result = Project(Path("/media"), Path("/work"))
    result.assets = [Asset("a", Path("a.mp4"), "video", AssetRole.CAMERA),
                     Asset("b", Path("b.mp4"), "video", AssetRole.CAMERA)]
    result.assets[1].group_id = "Bob-camera"
    result.placements = [SourcePlacement("a", 0, 0, 10), SourcePlacement("b", 0, 0, 10)]
    return result


def test_speaker_switches_preserve_timeline_and_drive_program_pieces() -> None:
    source = project()
    edit = speaker_camera_edit(source, EditMap("cut", (KeepRange(0, 4),)), [
        {"start": 0, "end": 2, "speaker": "Alice"},
        {"start": 2, "end": 4, "speaker": "Bob"},
    ], {"Alice": "a", "Bob": "Bob-camera"})
    assert edit.keep == (KeepRange(0, 2, "a"), KeepRange(2, 4, "b"))
    assert [piece.video.name for piece in program_pieces(source, edit)] == ["a.mp4", "b.mp4"]


def test_uncertain_and_short_turns_hold_previous_camera() -> None:
    edit = speaker_camera_edit(project(), EditMap("cut", (KeepRange(0, 4),)), [
        {"start": 0, "end": 0.2, "speaker": "Alice"},
        {"start": 0.2, "end": 0.4, "speaker": "Bob"},
        {"start": 0.4, "end": 2, "speaker": "overlap"},
        {"start": 2, "end": 3, "speaker": "unknown"},
    ], {"Alice": "a", "Bob": "b"}, min_shot_s=1)
    assert edit.keep == (KeepRange(0, 4, "a"),)


def test_uncovered_preferred_camera_falls_back_and_manual_selection_wins() -> None:
    source = project()
    source.placements[1] = SourcePlacement("b", 2, 0, 2)
    turns = [{"start": 0, "end": 4, "speaker": "Bob"}]
    edit = speaker_camera_edit(source, EditMap("cut", (KeepRange(0, 4),)),
                               turns, {"Bob": "b"}, min_shot_s=0)
    assert edit.keep == (KeepRange(0, 2, "a"), KeepRange(2, 4, "b"))
    manual = speaker_camera_edit(source, EditMap("cut", (KeepRange(0, 4, "a"),)),
                                 turns, {"Bob": "b"}, min_shot_s=0)
    assert manual.keep == (KeepRange(0, 4, "a"),)


def test_removed_ranges_are_not_reintroduced() -> None:
    edit = speaker_camera_edit(project(), EditMap("cut", (KeepRange(0, 1), KeepRange(3, 4))),
                               [], {"Alice": "a"})
    assert edit.keep == (KeepRange(0, 1, "a"), KeepRange(3, 4, "a"))


def test_invalid_camera_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown speaker camera"):
        speaker_camera_edit(project(), EditMap("cut", (KeepRange(0, 1),)), [], {"X": "missing"})


def wide_project() -> Project:
    result = project()
    result.assets.append(Asset("w", Path("w.mp4"), "video", AssetRole.CAMERA))
    result.placements.append(SourcePlacement("w", 0, 0, 10))
    return result


MAPPING = {"Alice": "a", "Bob": "b", WIDE_SHOT: "w"}


def test_overlap_goes_wide_only_when_a_wide_shot_exists() -> None:
    turns = [{"start": 0, "end": 2, "speaker": "Alice"},
             {"start": 2, "end": 4, "speaker": "overlap"}]
    edit = speaker_camera_edit(wide_project(), EditMap("cut", (KeepRange(0, 4),)),
                               turns, MAPPING, min_shot_s=1)
    assert edit.keep == (KeepRange(0, 2, "a"), KeepRange(2, 4, "w"))
    without = {key: value for key, value in MAPPING.items() if key != WIDE_SHOT}
    held = speaker_camera_edit(project(), EditMap("cut", (KeepRange(0, 4),)),
                               turns, without, min_shot_s=1)
    assert held.keep == (KeepRange(0, 4, "a"),)


def test_rapid_exchanges_use_wide_then_return_to_speaker() -> None:
    turns = [{"start": 0, "end": 3, "speaker": "Alice"},
             *({"start": 3 + i, "end": 4 + i, "speaker": "Bob" if i % 2 == 0 else "Alice"}
               for i in range(4)),
             {"start": 7, "end": 10, "speaker": "Bob"}]
    edit = speaker_camera_edit(wide_project(), EditMap("cut", (KeepRange(0, 10),)),
                               turns, MAPPING, policy=CameraPolicy(min_shot_s=1))
    cameras = [item.camera_id for item in edit.keep]
    assert cameras[0] == "a" and "w" in cameras and cameras[-1] == "b"
    assert sum(1 for left, right in zip(cameras, cameras[1:], strict=False) if left != right) <= 3


def test_long_monologue_gets_wide_insert_and_manual_choice_is_untouched() -> None:
    turns = [{"start": 0, "end": 10, "speaker": "Alice"}]
    policy = CameraPolicy(min_shot_s=1, max_shot_s=4, wide_insert_s=2)
    edit = speaker_camera_edit(wide_project(), EditMap("cut", (KeepRange(0, 10),)),
                               turns, MAPPING, policy=policy)
    assert edit.keep == (KeepRange(0, 4, "a"), KeepRange(4, 6, "w"),
                         KeepRange(6, 10, "a"))
    manual = speaker_camera_edit(wide_project(), EditMap("cut", (KeepRange(0, 10, "a"),)),
                                 turns, MAPPING, policy=policy)
    assert manual.keep == (KeepRange(0, 10, "a"),)


def test_cut_lead_switches_before_the_turn() -> None:
    edit = speaker_camera_edit(project(), EditMap("cut", (KeepRange(0, 4),)), [
        {"start": 0, "end": 2, "speaker": "Alice"},
        {"start": 2, "end": 4, "speaker": "Bob"},
    ], {"Alice": "a", "Bob": "b"}, policy=CameraPolicy(min_shot_s=0, cut_lead_s=0.2))
    assert edit.keep == (KeepRange(0, 1.8, "a"), KeepRange(1.8, 4, "b"))
