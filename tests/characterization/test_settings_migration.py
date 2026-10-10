"""Settings migration against the frozen legacy configuration schemas."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from studio.core.settings import SettingsError, load_settings
from tools.migrate_settings import main, migrate

ROOT = Path(__file__).resolve().parents[2]


def test_every_whispersync_field_is_classified() -> None:
    from whispersync.config import WhisperSyncConfig

    defaults = asdict(WhisperSyncConfig())
    defaults.pop("config_path", None)
    migration = migrate(whispersync=defaults)
    report = migration.report()
    unknown = [item for item in report["unmapped"] if item["planned"] == "unknown"]
    assert unknown == []
    assert len(report["migrated"]) + len(report["unmapped"]) == len(defaults)
    sync = migration.settings["sync"]
    assert sync["strategy"] == 3 and sync["master_crossfade_ms"] == 10.0
    assert migration.settings["roughcut"]["detect_retakes"] is False


def test_crossfade_disabled_in_whispersync_becomes_zero() -> None:
    migration = migrate(whispersync={"crossfade_enabled": False, "crossfade_ms": 25})
    assert migration.settings["sync"]["master_crossfade_ms"] == 0


def test_frozen_forge_config_migrates_to_valid_studio_settings(tmp_path: Path) -> None:
    source = yaml.safe_load((ROOT / "legacy" / "forge" / "config.yaml").read_text("utf-8"))
    migration = migrate(forge=source)
    report = migration.report()
    assert [item for item in report["unmapped"] if item["planned"].startswith("unknown")] == []
    settings = migration.settings
    assert settings["text"]["base_url"] == "http://127.0.0.1:11440"
    assert settings["llm"]["port"] == 11440
    assert settings["llm"]["roles"]["cleanup"] == {"model": "gemma4:26b"}
    assert settings["transcribe"]["language"] is None
    planned = {item["planned"] for item in report["unmapped"]}
    assert {"fetch", "batch", "reels"} <= planned
    assert settings["fetch"]["download"] == "audio"
    assert settings["fetch"]["exclude"] == ["PLoXa43IuWqDcGdAQbPvtJy0TzS8cWeIC6"]
    assert settings["batch"] == {"notify_when": "failure", "notify_webhook": "",
                                 "min_free_disk_gb": 5.0}
    reasons = {item["source"]: item["reason"] for item in report["unmapped"]}
    assert "uvf channel --limit" in reasons["forge:youtube.limit"]
    assert all(reason != "no Studio setting yet" for source, reason in reasons.items()
               if source.startswith(("forge:youtube.", "forge:autonomy.", "forge:cli.")))


def test_dependencies_are_reconciled_and_invalid_values_rejected() -> None:
    migration = migrate(forge={"video": {"smart_crop_face": True, "vertical_crop": False},
                               "proofread": {"enabled": False, "terms": {"enabled": True}}})
    assert migration.settings["reels"]["tracking"] is False
    assert migration.settings["text"]["term_check"] is False
    assert {item["target"] for item in migration.conflicts} == {
        "reels.tracking", "text.term_check",
    }
    with pytest.raises(SettingsError, match="migrated settings are invalid"):
        migrate(whispersync={"default_strategy": 7})


def test_cli_writes_loadable_settings_and_report(tmp_path: Path) -> None:
    whispersync = tmp_path / "ws.json"
    whispersync.write_text(json.dumps({"model": "medium", "detect_retakes": True,
                                       "gcc_eps": 1e-6}), encoding="utf-8")
    forge = tmp_path / "config.yaml"
    forge.write_text(yaml.safe_dump({"transcription": {"model": "large-v3"}}), "utf-8")
    output, report = tmp_path / "settings.yaml", tmp_path / "report.json"
    assert main(["--whispersync", str(whispersync), "--forge", str(forge),
                 "--output", str(output), "--report", str(report)]) == 0
    loaded = load_settings([output])
    assert loaded.transcribe.model == "large-v3" and loaded.roughcut.detect_retakes
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["conflicts"][0]["target"] == "transcribe.model"
    assert data["unmapped"][0]["source"] == "whispersync:gcc_eps"
    assert main(["--forge", str(forge), "--output", str(output)]) == 2
