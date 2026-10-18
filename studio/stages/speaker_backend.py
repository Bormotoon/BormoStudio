"""Isolated pyannote worker; model credentials stay in the environment."""

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studio.core.proc import run_logged
from studio.modules.manager import ModuleManager

WORKER = """
import json, os, sys
# pyannote.audio 3.x checkpoints are Lightning pickles; torch >= 2.6 (needed for
# Blackwell GPUs) loads weights only by default and rejects them. The weights
# come from the pinned pyannote repositories on Hugging Face.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
import torch
from pyannote.audio import Pipeline
model, device, audio, output = sys.argv[1:5]
known = {'num_speakers': int(sys.argv[5])} if len(sys.argv) > 5 and int(sys.argv[5]) else {}
pipeline = Pipeline.from_pretrained(model, use_auth_token=os.environ.get('HF_TOKEN'))
if pipeline is None:
    raise RuntimeError('diarization model unavailable; check model access and HF_TOKEN')
pipeline.to(torch.device(device))
embeddings = {}
try:
    annotation, centroids = pipeline(audio, return_embeddings=True, **known)
    for label, vector in zip(annotation.labels(), centroids):
        values = [float(value) for value in vector]
        if all(value == value for value in values):
            embeddings[str(label)] = values
except TypeError:
    annotation = pipeline(audio, **known)
turns = [{'start': segment.start, 'end': segment.end, 'speaker': str(speaker)}
         for segment, _, speaker in annotation.itertracks(yield_label=True)]
with open(output, 'w', encoding='utf-8') as stream:
    json.dump({'turns': turns, 'embeddings': embeddings}, stream)
"""


@dataclass(frozen=True, slots=True)
class Diarization:
    turns: list[dict[str, Any]]
    # Speaker centroid per label when the pipeline provides one.
    embeddings: dict[str, list[float]] = field(default_factory=dict)


def diarize(
    source: Path, directory: Path, *, model: str, device: str, num_speakers: int = 0
) -> Diarization:
    """pyannote turns of ``source``; ``num_speakers`` > 0 tells it how many people talk."""
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
                             model, device, str(audio), str(output), str(num_speakers)],
                            log_dir=directory / "logs", timeout=7200, keep_logs=True)
        if result.returncode:
            raise RuntimeError("pyannote worker failed; inspect speakers backend logs")
        payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("pyannote worker returned malformed output")
    turns, embeddings = payload.get("turns"), payload.get("embeddings", {})
    if not isinstance(embeddings, dict) or not all(
        isinstance(vector, list) and vector and all(isinstance(v, (int, float)) for v in vector)
        for vector in embeddings.values()
    ):
        raise ValueError("pyannote worker returned malformed embeddings")
    if not isinstance(turns, list) or not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("pyannote worker returned malformed turns")
    checked: list[dict[str, Any]] = turns
    for turn in checked:
        start, end = float(turn["start"]), float(turn["end"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError("invalid pyannote turn timing")
        if not isinstance(turn["speaker"], str) or not turn["speaker"]:
            raise ValueError("invalid pyannote speaker label")
    return Diarization(checked, {str(k): [float(v) for v in vector]
                                 for k, vector in embeddings.items()})
