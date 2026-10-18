from pathlib import Path

from studio.core.project import Project
from studio.core.settings import LATER_DEFAULTS, Settings, fingerprint_view
from studio.stages.base import select_settings
from studio.stages.scan import ScanStage, _media_paths
from studio.stages.sync import self_check_transcribe


def test_settings_added_later_leave_fingerprints_alone(tmp_path: Path) -> None:
    full = Settings().to_dict()
    before = {"scan": {k: v for k, v in full["scan"].items() if k != "extra_extensions"},
              "sync": {k: v for k, v in full["sync"].items()
                       if f"sync.{k}" not in LATER_DEFAULTS}}
    keys = ScanStage.settings_keys
    assert select_settings(full, keys) == select_settings(before, keys)
    assert fingerprint_view("sync", full["sync"]) == before["sync"]
    project = Project(tmp_path, tmp_path / "_studio")
    assert ScanStage().fingerprint(project, full) == ScanStage().fingerprint(project, before)
    changed = {**full, "scan": {**full["scan"], "extra_extensions": [".mts"]}}
    assert ScanStage().fingerprint(project, changed) != ScanStage().fingerprint(project, full)


def test_extra_extensions_and_the_self_check_mode(tmp_path: Path) -> None:
    for name in ("a.mov", "b.mts", "c.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in _media_paths(tmp_path)] == ["a.mov"]
    assert [p.name for p in _media_paths(tmp_path, frozenset({".mts"}))] == ["a.mov", "b.mts"]
    settings = {"transcribe": {"mode": "quality", "model": "m"}, "sync": {}}
    assert self_check_transcribe(settings)["mode"] == "quality"
    settings["sync"]["self_check_transcribe_mode"] = "fast"
    assert self_check_transcribe(settings) == {"mode": "fast", "model": "m"}
