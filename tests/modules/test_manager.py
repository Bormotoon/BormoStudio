import json
from pathlib import Path

import pytest

from studio.modules.manager import RECIPES, ModuleManager, module_names


def test_module_status_requires_interpreter_and_marker(tmp_path: Path) -> None:
    manager = ModuleManager(tmp_path)
    assert "vision" in module_names()
    assert not manager.installed("vision")
    interpreter = manager.interpreter("vision")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    assert not manager.installed("vision")
    marker = tmp_path / "vision" / "module.json"
    marker.write_text(json.dumps({"requirements": list(RECIPES["vision"].requirements)}),
                      encoding="utf-8")
    assert manager.installed("vision")


def test_module_built_from_an_older_recipe_is_installed_again(tmp_path: Path) -> None:
    manager = ModuleManager(tmp_path)
    interpreter = manager.interpreter("diarization")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    marker = tmp_path / "diarization" / "module.json"
    for stale in ("{}", "not json", json.dumps({"requirements": ["pyannote.audio>=3.1,<4"]})):
        marker.write_text(stale, encoding="utf-8")
        assert not manager.installed("diarization")


def test_unknown_module_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        ModuleManager(tmp_path).install("missing")
