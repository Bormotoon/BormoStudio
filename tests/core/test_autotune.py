import json
from pathlib import Path

import pytest

import studio.core.autotune as autotune
from studio.core.autotune import Measured, VramSampler, tune_llama, tune_whisper

PROBE = {"available": True, "cuda_devices": 1,
         "compute_types": ["float16", "int8_float16", "int8", "float32"]}


def _fake_whisper(speeds: dict[tuple[str, int], float | None],
                  memory: dict[tuple[str, int], int]):  # type: ignore[no-untyped-def]
    calls = []

    def run(command: list[str], timeout: float) -> Measured:
        request = json.loads(command[command.index("--bench-whisper") + 1])
        key = (request["compute_type"], request["batch_size"])
        calls.append(key)
        speed = speeds.get(key)
        if speed is None:
            return Measured(1, "", "CUDA failed: out of memory", 3.0, None)
        result = {"audio_s": 60.0, "transcribe_s": 60.0 / speed, "load_s": 2.0, "words": 90}
        return Measured(0, json.dumps(result), "", 60.0 / speed + 2, memory.get(key, 3000))
    return run, calls


def test_whisper_tuning_grows_batches_until_no_gain_or_failure(tmp_path: Path) -> None:
    speeds = {("float16", 4): 20.0, ("float16", 8): 30.0, ("float16", 16): 30.5,
              ("int8_float16", 4): 25.0, ("int8_float16", 8): None,
              ("int8", 4): 32.0, ("int8", 8): 33.0}
    run, calls = _fake_whisper(speeds, {})
    report = tune_whisper(tmp_path / "s.wav", 60.0, {"model": "large-v3"}, PROBE, run=run,
                          free_mb=12000)
    # float16 stops after 16 (no 5% gain); int8_float16 stops at its OOM.
    assert calls[:4] == [("float16", 4), ("float16", 8), ("float16", 16),
                         ("int8_float16", 4)]
    assert ("float16", 24) not in calls and ("int8_float16", 16) not in calls
    # int8 at 32x is penalized to 28.8 and loses to float16 at 30.5x.
    assert report["recommendations"] == {"transcribe": {
        "device": "cuda", "compute_type": "float16", "batch_size": 16}}
    failed = [t for t in report["trials"] if not t["ok"]]
    assert failed and "out of memory" in failed[0]["error"]


def test_whisper_trials_that_do_not_fit_free_memory_are_rejected(tmp_path: Path) -> None:
    speeds = {("float16", 4): 20.0, ("float16", 8): 30.0, ("int8_float16", 4): 18.0,
              ("int8", 4): 10.0}
    memory = {("float16", 8): 9000}
    run, calls = _fake_whisper(speeds, memory)
    report = tune_whisper(tmp_path / "s.wav", 60.0, {}, PROBE, run=run, free_mb=8000)
    assert report["recommendations"]["transcribe"]["batch_size"] == 4
    assert ("float16", 16) not in calls  # a bigger batch than one that did not fit


def test_cpu_only_machines_tune_cpu_types(tmp_path: Path) -> None:
    run, calls = _fake_whisper({("int8", 1): 2.0, ("float32", 1): 1.0}, {})
    report = tune_whisper(tmp_path / "s.wav", 60.0, {},
                          {"available": True, "cuda_devices": 0,
                           "compute_types": ["int8", "float32"]}, run=run)
    assert report["device"] == "cpu" and calls[0] == ("int8", 1)


def test_llama_tuning_scores_a_representative_job(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    monkeypatch.setattr(autotune.shutil, "which", lambda name: "/bin/llama-server")
    seen = []

    def trial(llm, executable, model_path, params, work_dir, *, prompt, timeout_s):  # type: ignore[no-untyped-def]
        seen.append(params)
        if params["n_gpu_layers"] == 999 and params["batch_size"] == 2048:
            raise RuntimeError("llama-server exited while loading (out of memory?)")
        gpu = params["n_gpu_layers"] == 999
        return ({"prompt_per_second": 900.0 if gpu else 300.0,
                 "predicted_per_second": 40.0 if gpu else 12.0}, 14000 if gpu else 6000)

    report = tune_llama({"model_path": str(model)}, work_dir=tmp_path, trial=trial,
                        free_mb=15000)
    assert len(seen) == 6
    assert report["recommendations"] == {"llm": {"n_gpu_layers": 999, "batch_size": 1024,
                                                 "ubatch_size": 512}}
    assert report["best"]["details"]["job_s"] == pytest.approx(3000 / 900 + 800 / 40)
    tight = tune_llama({"model_path": str(model)}, work_dir=tmp_path, trial=trial,
                       free_mb=10000)
    assert tight["recommendations"]["llm"]["n_gpu_layers"] is None
    with pytest.raises(ValueError):
        tune_llama({"model_path": str(tmp_path / "missing.gguf")}, work_dir=tmp_path)


def test_vram_sampler_keeps_the_peak_of_its_process() -> None:
    readings = iter([{1: 100, 2: 9000}, {1: 700}, {1: 300}] + [{1: 0}] * 1000)
    sampler = VramSampler(1, interval_s=0.001, sample=lambda: next(readings))
    import time

    time.sleep(0.05)
    assert sampler.stop() == 700


def test_tune_command_writes_report_and_applies(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    from studio.cli.main import main
    from studio.core.settings import load_settings

    recommendation = {"transcribe": {"device": "cuda", "compute_type": "float16",
                                     "batch_size": 24}}
    monkeypatch.setattr(autotune, "tune_project", lambda *a, **k: {
        "schema_version": 1, "recommendations": recommendation})
    assert main(["tune", str(tmp_path), "--whisper", "--apply"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is True
    assert json.loads((tmp_path / "_studio" / "tuning.json").read_text())["recommendations"]
    assert load_settings([tmp_path / "_studio" / "settings.yaml"]).transcribe.batch_size == 24
