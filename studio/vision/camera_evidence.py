"""Per-window face and active-speaker evidence for camera ↔ speaker binding.

Runs inside the managed vision interpreter. For every requested window of one
camera file it reports how many people stay in frame, how large the biggest
face is, and (with Light-ASD) how strongly the most active face is speaking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from studio.vision import active_speaker
from studio.vision.face_track import (
    GpuFaceDetector,
    TrackingSettings,
    _cuda_ready,
    probe_size,
    scan_clip,
    speaking_by_sample,
)


def window_evidence(
    video: Path, windows: list[tuple[float, float]], settings: TrackingSettings,
) -> dict[str, Any]:
    src_w, src_h = probe_size(video)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"cannot read video size of {video}")
    if settings.device.startswith("cuda") and not _cuda_ready():
        raise RuntimeError("camera binding needs a usable CUDA GPU for face detection")
    use_asd = settings.active_speaker and active_speaker.asd_available(download=False)
    detector = GpuFaceDetector(src_w, src_h, device=settings.device)
    step = max(1, int(round(active_speaker.VIDEO_FPS / settings.detect_fps)))
    results = []
    for start, end in windows:
        samples, _cuts, tracks = scan_clip(
            video, start, end, detector, settings, collect_crops=use_asd
        )
        n = len(samples)
        strong = [track for track in tracks if n and len(track.idx) >= 0.5 * n]
        heights = [float(np.median([face.height_px for face in track.faces])) / src_h
                   for track in strong]
        speaking = None
        if use_asd and strong:
            scores = speaking_by_sample(
                video, start, end, strong, n_samples=n, frames_per_sample=step,
                device=settings.device,
            )
            means = [float(np.nanmean(values)) for values in scores.values()
                     if np.isfinite(values).any()]
            speaking = max(means) if means else None
        results.append({
            "start": start, "end": end, "faces": len(strong),
            "face_height": max(heights) if heights else 0.0, "speaking": speaking,
        })
    return {"windows": results, "active_speaker": use_asd}
