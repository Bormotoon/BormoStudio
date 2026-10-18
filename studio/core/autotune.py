"""Measured Whisper and llama tuning on this machine.

Each candidate runs in its own process, so an out-of-memory failure ends that
trial and not the application. Wall time comes from the trial itself; GPU
memory is the trial process's peak as reported by nvidia-smi. The winner is
the fastest trial that fits in the free GPU memory measured before tuning
minus a headroom. Results are recommendations; applying them is explicit.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from studio.core.launch import python_command

HEADROOM_MB = 1024
# int8 Whisper decodes slightly worse; it has to be clearly faster to win.
INT8_PENALTY = 0.9
# A representative LLM job: a long transcript chunk and a JSON answer.
LLM_PROMPT_TOKENS = 3000
LLM_OUTPUT_TOKENS = 800


@dataclass(slots=True)
class Trial:
    params: dict[str, Any]
    ok: bool
    seconds: float | None = None
    score: float | None = None
    peak_vram_mb: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Measured:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    peak_vram_mb: int | None


def gpu_process_memory() -> dict[int, int] | None:
    """MiB of GPU memory per process id, or None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = usage.get(int(parts[0]), 0) + int(parts[1])
    return usage


def free_vram_mb() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class VramSampler:
    """Poll one process's GPU memory in the background and keep the peak."""

    def __init__(self, pid: int, interval_s: float = 0.5,
                 sample: Callable[[], dict[int, int] | None] = gpu_process_memory) -> None:
        import threading

        self.pid, self.interval_s, self.sample = pid, interval_s, sample
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            usage = self.sample()
            if usage is not None and self.pid in usage:
                self.peak = max(self.peak or 0, usage[self.pid])
            self._stop.wait(self.interval_s)

    def stop(self) -> int | None:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak


def run_measured(command: list[str], timeout: float) -> Measured:
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    sampler = VramSampler(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        stderr += "\ntrial timed out"
    finally:
        peak = sampler.stop()
    return Measured(process.returncode, stdout, stderr, time.monotonic() - started, peak)


def _fits(trial: Trial, free_mb: int | None, headroom_mb: int) -> bool:
    return free_mb is None or trial.peak_vram_mb is None or (
        trial.peak_vram_mb <= free_mb - headroom_mb)


def _best(trials: list[Trial], free_mb: int | None, headroom_mb: int) -> Trial | None:
    usable = [trial for trial in trials if trial.ok and trial.score is not None
              and _fits(trial, free_mb, headroom_mb)]
    return max(usable, key=lambda t: (t.score or 0.0, -(t.peak_vram_mb or 0)), default=None)


# --- Whisper -----------------------------------------------------------------

def whisper_benchmark(request: dict[str, Any]) -> dict[str, Any]:
    """Transcribe ``request["audio"]`` once with the given settings (trial process)."""
    from studio.stages.transcribe import TranscribeSettings, whisper_backend

    settings = TranscribeSettings.from_mapping({
        "model": request["model"], "device": request["device"], "language":
        request.get("language"), "compute_type": request["compute_type"],
        "batch_size": request["batch_size"], "mode": "fast", "use_cache": False,
    })
    backend = whisper_backend(settings)
    started = time.monotonic()
    load = getattr(backend, "_ensure_model", None)
    if load is not None:
        load()
    loaded = time.monotonic()
    transcript = backend.transcribe(Path(request["audio"]), settings)
    finished = time.monotonic()
    return {"load_s": loaded - started, "transcribe_s": finished - loaded,
            "audio_s": float(request["audio_s"]), "words": len(transcript.words)}


def whisper_candidates(probe: dict[str, Any]) -> tuple[str, list[str], list[int]]:
    if probe.get("mlx"):
        # mlx-whisper has one precision and no batched pipeline: one measurement.
        return "mlx", ["float16"], [1]
    cuda = bool(probe.get("available") and probe.get("cuda_devices"))
    supported = set(probe.get("compute_types", []))
    preferred = ["float16", "int8_float16", "int8"] if cuda else ["int8", "float32"]
    types = [value for value in preferred if value in supported] or preferred[:1]
    return ("cuda" if cuda else "cpu"), types, ([4, 8, 16, 24, 32] if cuda else [1, 4, 8])


def tune_whisper(
    sample: Path, sample_s: float, transcribe: dict[str, Any], probe: dict[str, Any],
    *, run: Callable[[list[str], float], Measured] = run_measured,
    free_mb: int | None = None, headroom_mb: int = HEADROOM_MB,
) -> dict[str, Any]:
    device, types, batches = whisper_candidates(probe)
    trials: list[Trial] = []
    for compute_type in types:
        best_for_type = 0.0
        for batch in batches:
            params = {"device": device, "compute_type": compute_type, "batch_size": batch}
            request = {**params, "model": transcribe.get("model", "large-v3"),
                       "language": transcribe.get("language"), "audio": str(sample),
                       "audio_s": sample_s}
            measured = run(python_command("studio.cli.main",
                                          ["compute", "--bench-whisper", json.dumps(request)]),
                           max(600.0, sample_s * 20))
            trial = Trial(params, False, measured.seconds, peak_vram_mb=measured.peak_vram_mb)
            try:
                if measured.returncode:
                    raise ValueError(measured.stderr.strip()[-500:] or "trial failed")
                result = json.loads(measured.stdout)
                speed = result["audio_s"] / max(result["transcribe_s"], 1e-6)
                penalty = INT8_PENALTY if compute_type.startswith("int8") else 1.0
                trial.ok, trial.score, trial.details = True, speed * penalty, result
            except (ValueError, KeyError) as exc:
                trial.error = str(exc)
            trials.append(trial)
            if not trial.ok or not _fits(trial, free_mb, headroom_mb):
                break  # a larger batch needs even more memory
            # Stop growing the batch once it no longer pays off.
            if trial.score is not None and trial.score < best_for_type * 1.05:
                break
            best_for_type = max(best_for_type, trial.score or 0.0)
    best = _best(trials, free_mb, headroom_mb)
    return {
        "device": device, "sample_s": sample_s, "free_vram_mb": free_mb,
        "headroom_mb": headroom_mb, "trials": [asdict(trial) for trial in trials],
        "best": asdict(best) if best else None,
        "recommendations": {"transcribe": best.params} if best else {},
    }


# --- llama -------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def llama_candidates(llm: dict[str, Any]) -> list[dict[str, Any]]:
    layers = [None, 999] if llm.get("n_gpu_layers") is None else [llm["n_gpu_layers"], None]
    return [{"n_gpu_layers": count, "batch_size": batch, "ubatch_size": ubatch}
            for count in layers for batch, ubatch in ((2048, 512), (1024, 512), (512, 256))]


def llama_trial(
    llm: dict[str, Any], executable: str, model: Path, params: dict[str, Any],
    work_dir: Path, *, prompt: str, timeout_s: float,
) -> tuple[dict[str, Any], int | None]:
    """Start an owned server with ``params``, time one completion, stop it."""
    from studio.llm.server import LlamaServer, server_arguments

    port = _free_port()
    settings = {**llm, **params}
    server = LlamaServer(Path(executable), model, work_dir, port,
                         arguments=server_arguments(settings, executable),
                         log_file=work_dir / "llama-tune.log")
    server.start()
    assert server.process is not None
    sampler = VramSampler(server.process.pid)
    try:
        deadline = time.monotonic() + float(llm.get("startup_timeout_s", 120))
        while True:
            if server.process.poll() is not None:
                raise RuntimeError("llama-server exited while loading (out of memory?)")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("llama-server did not become ready")
            time.sleep(0.2)
        body = {"prompt": prompt, "n_predict": 128, "temperature": 0, "cache_prompt": False}
        request = Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            timings = json.load(response).get("timings", {})
    finally:
        peak = sampler.stop()
        server.stop()
    return timings, peak


def tune_llama(
    llm: dict[str, Any], *, work_dir: Path,
    trial: Callable[..., tuple[dict[str, Any], int | None]] = llama_trial,
    free_mb: int | None = None, headroom_mb: int = 512,
) -> dict[str, Any]:
    model = Path(str(llm.get("model_path", ""))).expanduser()
    executable = shutil.which(str(llm.get("executable", "llama-server")))
    if not model.is_file() or executable is None:
        raise ValueError("llama tuning needs llm.model_path and an llama-server executable")
    # ~4 characters per token: a prompt about the size of a real chunk.
    prompt = ("Episode transcript excerpt. " * (LLM_PROMPT_TOKENS // 5))[: LLM_PROMPT_TOKENS * 4]
    trials: list[Trial] = []
    for params in llama_candidates(llm):
        started = time.monotonic()
        result = Trial(params, False)
        try:
            timings, peak = trial(llm, executable, model, params, work_dir, prompt=prompt,
                                  timeout_s=float(llm.get("first_token_timeout_s", 120)) + 120)
            prompt_tps = float(timings["prompt_per_second"])
            output_tps = float(timings["predicted_per_second"])
            job_s = LLM_PROMPT_TOKENS / prompt_tps + LLM_OUTPUT_TOKENS / output_tps
            result.ok, result.score, result.peak_vram_mb = True, 1 / job_s, peak
            result.details = {"prompt_tokens_per_s": prompt_tps,
                              "output_tokens_per_s": output_tps, "job_s": job_s}
        except (OSError, RuntimeError, ValueError, KeyError, ZeroDivisionError) as exc:
            result.error = str(exc)
        result.seconds = time.monotonic() - started
        trials.append(result)
    best = _best(trials, free_mb, headroom_mb)
    return {
        "model": str(model), "free_vram_mb": free_mb, "headroom_mb": headroom_mb,
        "job": {"prompt_tokens": LLM_PROMPT_TOKENS, "output_tokens": LLM_OUTPUT_TOKENS},
        "trials": [asdict(item) for item in trials],
        "best": asdict(best) if best else None,
        "recommendations": {"llm": best.params} if best else {},
    }


def extract_sample(source: Path, output: Path, seconds: float) -> float:
    """``seconds`` of 16 kHz mono speech from the middle of ``source``."""
    from studio.core.media import probe

    duration = probe(source).duration_s or seconds
    length = min(seconds, duration)
    start = max(0.0, (duration - length) / 2)
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t",
         f"{length:.3f}", "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
         str(output)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"cannot extract a tuning sample: {result.stderr[-500:]}")
    return length


def tune_project(
    assets: list[Path], settings: dict[str, Any], work_dir: Path,
    *, whisper: bool, llama: bool, sample_s: float = 60.0,
) -> dict[str, Any]:
    from studio.core.capabilities import inspect_compute

    report: dict[str, Any] = {"schema_version": 1, "created": time.time(),
                              "recommendations": {}}
    free = free_vram_mb()
    if whisper:
        if not assets:
            raise ValueError("Whisper tuning needs an audio or video file in the project")
        probe = inspect_compute()["whisper"]
        with tempfile.TemporaryDirectory(prefix="bstudio-tune-") as folder:
            sample = Path(folder) / "sample.wav"
            length = extract_sample(assets[0], sample, sample_s)
            report["whisper"] = tune_whisper(sample, length, settings.get("transcribe", {}),
                                             probe, free_mb=free)
        report["recommendations"].update(report["whisper"]["recommendations"])
    if llama:
        report["llama"] = tune_llama(settings.get("llm", {}), work_dir=work_dir / "llama",
                                     free_mb=free)
        report["recommendations"].update(report["llama"]["recommendations"])
    return report
