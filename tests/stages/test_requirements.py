import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from studio.gui.main_window import MainWindow
from studio.stages.base import Requirement
from studio.stages.requirements import missing_requirements


def test_missing_requirements_name_install_targets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BSTUDIO_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("studio.modules.manager.user_data_path", lambda name: tmp_path)
    found = missing_requirements([
        Requirement("binary", "definitely-not-a-binary-uvf", "test"),
        Requirement("binary", "definitely-not-a-binary-uvf", "duplicate"),
        Requirement("package", "json", "stdlib is present"),
        Requirement("module", "vision", "tracking", optional=True),
        Requirement("model", "yunet", "faces"),
        Requirement("llama", "local", "external endpoints are not probed"),
    ], {})
    assert [(item.kind, item.install) for item in found] == [
        ("binary", None), ("module", ("module", "vision")), ("model", ("model", "yunet")),
    ]
    assert found[1].optional
    managed = missing_requirements(
        [Requirement("llama", "local", "text")],
        {"llm": {"managed": True, "executable": "python3", "model_path": "/missing.gguf"}},
    )
    assert managed[0].install == ("model", "gguf-gemma-4-e4b-it-q4")


def test_plan_page_navigates_to_the_missing_module() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._plan_event({"stages": [
        {"id": "speakers", "decision": "run", "reason": "", "reuse": False, "missing": [
            {"value": "diarization", "hint": "install it", "optional": False,
             "install": ["module", "diarization"]},
        ]},
    ]})
    assert "missing: diarization" in window.plan_label.text()
    assert window.install_button.isEnabled()
    window.install_button.click()
    assert window.pages.currentWidget() is window.modules_page
    assert window.modules_page.module.currentText() == "diarization"
    assert app is not None
