"""Bounded backend probes and conservative project compute recommendations."""

import json
import shutil
import subprocess
import sys
from typing import Any

from studio.core.launch import python_command


def whisper_probe() -> dict:
    from studio.stages.transcribe import mlx_available

    if mlx_available():
        return {"available": True, "cuda_devices": 0, "mlx": True,
                "compute_types": ["float16"]}
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        supported = sorted(ctranslate2.get_supported_compute_types("cuda" if count else "cpu"))
        return {"available": True, "cuda_devices": count, "compute_types": supported}
    except Exception as exc:
        return {"available": False, "cuda_devices": 0, "error": str(exc)}


def inspect_compute() -> dict:
    from studio.stages.program_encoder import select_encoder

    diagnostics = {}
    try:
        result = subprocess.run(
            python_command("studio.cli.main", ["compute", "--probe-whisper"]),
            capture_output=True,
            text=True,
            timeout=30,
        )
        whisper: dict[str, Any] = (
            json.loads(result.stdout)
            if result.returncode == 0
            else {"available": False, "cuda_devices": 0, "error": result.stderr[-2000:]}
        )
        if not isinstance(whisper, dict):
            raise ValueError("whisper probe did not return an object")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        whisper = {"available": False, "cuda_devices": 0, "error": str(exc)}
    cuda = bool(whisper.get("available") and whisper.get("cuda_devices", 0))
    mlx = bool(whisper.get("mlx"))
    types = whisper.get("compute_types", [])
    preferred = ("float16", "int8_float16", "float32") if cuda else ("int8", "float32")
    compute_type = next((value for value in preferred if value in types), "auto")
    try:
        choice = select_encoder("auto")
        encoder = {"h264_nvenc": "nvenc", "h264_videotoolbox": "videotoolbox"}.get(
            choice.codec, "cpu")
        diagnostics["encoder"] = choice.reason
    except (OSError, RuntimeError) as exc:
        encoder = "cpu"
        diagnostics["encoder"] = str(exc)
    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "binaries": {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")},
        "whisper": whisper,
        "diagnostics": diagnostics,
        "recommendations": {
            "transcribe": {
                "device": "cuda" if cuda else "mlx" if mlx else "cpu",
                "compute_type": "auto" if mlx else compute_type,
                "batch_size": 8 if cuda else 1,
            },
            "program": {"encoder": encoder},
        },
    }
