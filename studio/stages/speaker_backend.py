"""Isolated pyannote worker; model credentials stay in the environment."""

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from studio.core.proc import run_logged
from studio.modules.manager import ModuleManager

WORKER = """
import json, os, sys
import torch
from pyannote.audio import Pipeline
model, device, audio, output = sys.argv[1:]
pipeline = Pipeline.from_pretrained(model, use_auth_token=os.environ.get('HF_TOKEN'))
if pipeline is None:
    raise RuntimeError('diarization model unavailable; check model access and HF_TOKEN')
pipeline.to(torch.device(device))
annotation = pipeline(audio)
turns = [{'start': segment.start, 'end': segment.end, 'speaker': str(speaker)}
         for segment, _, speaker in annotation.itertracks(yield_label=True)]
with open(output, 'w', encoding='utf-8') as stream:
    json.dump(turns, stream)
"""


def diarize(
    source: Path, directory: Path, *, model: str, device: str
) -> list[dict[str, Any]]:
    manager = ModuleManager()
    if not manager.installed("diarization"):
        raise RuntimeError("diarization module is missing; install the diarization module")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".diarize-", dir=directory) as temporary:
        audio = Path(temporary) / "input.wav"
        output = Path(temporary) / "turns.json"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", str(audio)],
                       capture_output=True, check=True)
        result = run_logged([str(manager.interpreter("diarization")), "-c", WORKER,
                             model, device, str(audio), str(output)],
                            log_dir=directory / "logs", timeout=7200, keep_logs=True)
        if result.returncode:
            raise RuntimeError("pyannote worker failed; inspect speakers backend logs")
        turns = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(turns, list) or not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("pyannote worker returned malformed turns")
    checked: list[dict[str, Any]] = turns
    for turn in checked:
        start, end = float(turn["start"]), float(turn["end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("invalid pyannote turn timing")
        if not isinstance(turn["speaker"], str) or not turn["speaker"]:
            raise ValueError("invalid pyannote speaker label")
    return checked
