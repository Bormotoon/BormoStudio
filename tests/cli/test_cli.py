import json
from pathlib import Path

import pytest

from studio.cli import main as cli_main
from studio.cli.main import main
from studio.core.project import Project, StageManifest
from studio.core.settings import load_settings
from studio.stages.prepare import PrepareStage
from tools.make_fixtures import generate


def test_scan_and_plan_fixture_through_workers(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = generate(tmp_path / "fixture")
    assert main(["scan", str(source), "--json"]) == 0
    scan = json.loads(capsys.readouterr().out)
    assert len(scan["assets"]) == 4
    assert main(["plan", str(source), "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [stage["id"] for stage in plan["stages"]] == [
        "scan",
        "prepare",
        "transcribe",
        "sync",
        "timeline",
        "speakers",
        "roughcut",
        "text",
        "program",
        "reels",
        "reel_render",
        "export",
        "program_subtitles",
    ]
    assert all(stage["reuse"] for stage in plan["stages"][:2])
    assert plan["stages"][2]["decision"] == "run"


def test_overrides_reach_worker_without_rewriting_project_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = generate(tmp_path / "fixture")
    work = source / "_studio"
    work.mkdir()
    settings_path = work / "settings.yaml"
    original = "# Project defaults\nsync:\n  mode: auto\n"
    settings_path.write_text(original, encoding="utf-8")

    assert main(["plan", str(source), "--json", "--set", "sync.mode=camera"]) == 0
    plan = json.loads(capsys.readouterr().out)
    requirements_path = work / "stages" / "prepare" / "requirements.json"
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    assert requirements["transcripts"] == [
        "camera-a-dji-0001", "camera-b-gx010024", "camera-b-gx020024",
    ]
    project = Project.load(work / "project.json")
    effective = load_settings([settings_path], ["sync.mode=camera"]).to_dict()
    manifest = StageManifest.load(work / "manifests" / "prepare.json")
    assert manifest.fingerprint == PrepareStage().fingerprint(project, effective)
    assert manifest.inputs["effective_settings"] == effective
    assert plan["stages"][1]["reuse"]
    assert settings_path.read_text(encoding="utf-8") == original
    assert not list(work.glob(".settings-*"))

    assert main(["plan", str(source), "--json"]) == 0
    capsys.readouterr()
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    assert len(requirements["transcripts"]) == 4
    assert settings_path.read_text(encoding="utf-8") == original


def test_invalid_override_returns_usage_error_from_console_entry_point(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["plan", str(tmp_path), "--set", "sync.mode=invalid", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "sync.mode" in captured.err
    assert not (tmp_path / "_studio" / "manifests").exists()


def test_modules_command_lists_and_installs_isolated_modules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeManager:
        installed_names = {"vision"}
        installed_request: str | None = None

        def installed(self, name: str) -> bool:
            return name in self.installed_names

        def interpreter(self, name: str) -> Path:
            return tmp_path / name / "python"

        def install(self, name: str, *, progress=None) -> Path:
            self.installed_request = name
            return self.interpreter(name)

    manager = FakeManager()
    monkeypatch.setattr(cli_main, "ModuleManager", lambda: manager)
    monkeypatch.setattr(cli_main, "module_names", lambda: ("vision", "youtube"))

    assert main(["modules"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["modules"] == [
        {"name": "vision", "installed": True, "interpreter": str(tmp_path / "vision" / "python")},
        {
            "name": "youtube",
            "installed": False,
            "interpreter": str(tmp_path / "youtube" / "python"),
        },
    ]

    assert main(["modules", "install", "youtube"]) == 0
    installed = json.loads(capsys.readouterr().out)
    assert manager.installed_request == "youtube"
    assert installed == {"name": "youtube", "interpreter": str(tmp_path / "youtube" / "python")}


def test_material_command_persists_overrides_through_rescan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = generate(tmp_path / "fixture")
    assert main(["scan", str(source), "--json"]) == 0
    scanned = json.loads(capsys.readouterr().out)
    asset_id = next(asset["id"] for asset in scanned["assets"] if asset["role"] == "camera")

    assert (
        main(
            [
                "material",
                str(source),
                "--asset",
                asset_id,
                "--role",
                "ignore",
                "--group",
                "wide",
                "--device",
                "Manual camera",
                "--json",
            ]
        )
        == 0
    )
    changed = json.loads(capsys.readouterr().out)
    assert changed == {
        "id": asset_id,
        "role": "ignore",
        "group_id": "wide",
        "device": "Manual camera",
    }

    assert main(["scan", str(source), "--json"]) == 0
    capsys.readouterr()
    project = Project.load(source / "_studio" / "project.json")
    asset = next(item for item in project.assets if item.id == asset_id)
    assert asset.role.value == "ignore"
    assert asset.group_id == "wide"
    assert asset.device == "Manual camera"


def test_batch_runs_each_project_and_reports_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = generate(tmp_path / "first")
    second = generate(tmp_path / "second")

    assert main(["batch", str(first), str(second), "--only", "scan", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ok"
    assert report["projects"] == [
        {"source": str(first), "status": "ok"},
        {"source": str(second), "status": "ok"},
    ]
    assert (first / "_studio" / "report.json").is_file()
    assert (second / "_studio" / "report.json").is_file()


def test_fetch_command_creates_a_scanned_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "download"

    def download(url: str, directory: Path, cancelled=None, options=None) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
        media = directory / "source.mp4"
        info = directory / "source.info.json"
        media.write_bytes(b"media")
        info.write_text("{}", encoding="utf-8")
        assert url == "https://example.invalid/video"
        return media, info

    def discover(directory: Path, *, include_prepare: bool, overrides=None) -> Project:
        assert not include_prepare
        project = Project(directory, directory / "_studio")
        project.save(project.work_dir / "project.json")
        return project

    monkeypatch.setattr(cli_main, "fetch_to_directory", download)
    monkeypatch.setattr(cli_main, "_run_discovery", discover)

    assert main(["fetch", "https://example.invalid/video", str(destination), "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["url"] == "https://example.invalid/video"
    assert (destination / "_studio" / "stages" / "fetch" / "report.json").is_file()


def test_pre_cancelled_pipeline_reports_cancellation(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    cancel = tmp_path / "cancel"
    cancel.touch()
    assert main(["run", str(source), "--cancel-file", str(cancel)]) == 130
    report = json.loads((source / "_studio" / "report.json").read_text())
    assert report["status"] == "cancelled"
    assert not (source / "_studio" / ".studio-run.lock").exists()
    assert "cancelled" in capsys.readouterr().out


def test_discovery_and_batch_pre_cancel(tmp_path: Path, capsys) -> None:
    cancel = tmp_path / "cancel"
    cancel.touch()
    source = tmp_path / "material"
    source.mkdir()
    for command in ("scan", "plan"):
        assert main([command, str(source), "--cancel-file", str(cancel)]) == 130
        capsys.readouterr()
    other = tmp_path / "not-started"
    assert main(["batch", str(source), str(other), "--cancel-file", str(cancel), "--json"]) == 130
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "cancelled"
    assert all(item["status"] == "not_started" for item in report["projects"])
    assert not other.exists()


def test_reel_command_edits_and_clears_overrides(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    import json as _json

    assert main(["reel", str(tmp_path), "--reel", "c1", "--framing", "crop",
                 "--crop-x", "0.3", "--title", "Hook"]) == 0
    assert main(["reel", str(tmp_path), "--reel", "c1", "--tracking"]) == 0
    capsys.readouterr()
    assert main(["reel", str(tmp_path)]) == 0
    assert _json.loads(capsys.readouterr().out) == {"reels": {"c1": {
        "framing": "crop", "crop_x": 0.3, "title": "Hook", "tracking": True}}}
    assert main(["reel", str(tmp_path), "--reel", "c1", "--crop-x", "4"]) == 2
    capsys.readouterr()
    assert main(["reel", str(tmp_path), "--reel", "c1", "--clear"]) == 0
    assert _json.loads(capsys.readouterr().out) == {"reels": {}}
