import json
from pathlib import Path

import pytest

from studio.cli.main import main
from studio.core.presets import apply_preset, list_presets, load_preset
from studio.core.settings import load_settings


def test_every_packaged_preset_is_valid(tmp_path: Path) -> None:
    assert {"roughcut-balanced", "roughcut-gentle", "roughcut-tight"} <= set(list_presets())
    for name in list_presets():
        work = tmp_path / name
        work.mkdir()
        apply_preset(work, name)
        load_settings([work / "settings.yaml"])


def test_preset_merges_without_touching_other_keys(tmp_path: Path) -> None:
    (tmp_path / "settings.yaml").write_text(
        "sync:\n  mode: simple\nroughcut:\n  llm: true\n", encoding="utf-8"
    )
    apply_preset(tmp_path, "roughcut-tight")
    settings = load_settings([tmp_path / "settings.yaml"])
    assert settings.sync.mode == "simple"
    assert settings.roughcut.llm is True
    assert settings.roughcut.pause_min_s == load_preset("roughcut-tight")["roughcut"]["pause_min_s"]
    with pytest.raises(ValueError, match="unknown preset"):
        apply_preset(tmp_path, "missing")


def test_cli_lists_and_applies_presets(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["preset"]) == 0
    assert "roughcut-gentle" in capsys.readouterr().out.split()
    assert main(["preset", str(tmp_path), "--apply", "roughcut-gentle"]) == 0
    assert json.loads(capsys.readouterr().out)["roughcut"]["pause_min_s"] == 1.5
    assert load_settings([tmp_path / "_studio" / "settings.yaml"]).roughcut.pause_min_s == 1.5
    assert main(["preset", str(tmp_path), "--apply", "nope"]) == 2
