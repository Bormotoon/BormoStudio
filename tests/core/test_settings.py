from pathlib import Path

import pytest

from studio.core.settings import SettingsError, load_settings


def test_layers_and_cli_overrides_merge_by_section(tmp_path: Path) -> None:
    common = tmp_path / "common.yaml"
    project = tmp_path / "project.yaml"
    common.write_text("sync:\n  mode: simple\ntranscribe:\n  batch_size: 8\n")
    project.write_text("sync:\n  max_drift_ms: 12\n")
    settings = load_settings(
        [common, project], ["sync.mode=complex", "roughcut.enabled=false"]
    )
    assert settings.sync.mode == "complex"
    assert settings.sync.max_drift_ms == 12
    assert settings.transcribe.batch_size == 8
    assert not settings.roughcut.enabled


def test_invalid_settings_fail_before_work(tmp_path: Path) -> None:
    cases = [
        ("sync:\n  strategy: 4\n", "strategy"),
        ("roughcut:\n  pause_min_s: .nan\n", "finite"),
        ("unknown:\n  value: true\n", "unknown settings sections"),
        ("transcribe:\n  batch_size: false\n", "batch_size"),
        ("sync:\n  self_check: repair\n  voice_enhance: denoise\n", "requires"),
        ("program:\n  fps: '0/1'\n", "program.fps"),
        ("program:\n  fps: 'garbage'\n", "program.fps"),
        ("reels:\n  cleanup_max_candidates: -1\n", "cleanup_max_candidates"),
        ("reels:\n  scout_quote_ratio: 0.9\n", "must not exceed"),
        ("reels:\n  scoring_weights: {charisma: 2}\n", "scoring_weights"),
        ("reels:\n  max_overlap_ratio: 2\n", "max_overlap_ratio"),
        ("reels:\n  chunk_overlap_s: -5\n", "chunk_overlap_s"),
        ("reels:\n  min_duration_s: 60\n  max_duration_s: 30\n", "max_duration_s"),
        ("reels:\n  judge_max_candidates: false\n", "judge_max_candidates"),
        ("reels:\n  judge_max_candidates: 1.5\n", "judge_max_candidates"),
        ("reels:\n  retry_budget: -1\n", "retry_budget"),
        ("reels:\n  json_retries: true\n", "json_retries"),
        ("reels:\n  request_timeout_s: .nan\n", "request_timeout_s"),
        ("reels:\n  request_timeout_s: 0\n", "request_timeout_s"),
        ("reels:\n  retry_backoff_s: -1\n", "retry_backoff_s"),
        ("reels:\n  retry_backoff_s: 31\n", "retry_backoff_s"),
    ]
    path = tmp_path / "settings.yaml"
    for content, message in cases:
        path.write_text(content)
        with pytest.raises(SettingsError, match=message):
            load_settings([path])


def test_verified_repair_mode_is_available() -> None:
    assert load_settings([], ["sync.self_check=repair"]).sync.self_check == "repair"


@pytest.mark.parametrize("encoder", ["cpu", "libx264"])
def test_program_rate_and_cpu_aliases(encoder: str) -> None:
    settings = load_settings([], [f"program.encoder={encoder}", 'program.fps="30000/1001"'])
    assert settings.program.fps == "30000/1001"
    assert settings.program.encoder == encoder
