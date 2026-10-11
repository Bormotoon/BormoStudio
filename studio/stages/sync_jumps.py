"""Jumps of the recorder clock, shared by every camera that overlaps them.

A recorder that silently loses or duplicates samples shifts everything after
that moment by the same amount for every camera. On sony_run two independent
cameras moved at the same recorder time by the same step (≈ 843 s: dji-0921
−517 ms and mvi-0845 −515 ms), while each camera corrected only what its own
sound showed. So the steps the acoustic check finds are collected per
recorder, a jump seen by one camera is checked against the others (a camera
whose own sound covers both sides without a step there says it is the first
camera's own glitch), the text anchors of each camera confirm the large ones,
and clips that had no acoustic model of their own are given the accepted
jumps of their recorder: anchors on the jump-corrected line, one regime per
side of each jump, so the piece plan absorbs every jump in a pause.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from studio.stages.sync_drift import JUMP_TOKEN, DriftCheck, anchor_regime
from studio.stages.sync_matcher import ransac_linear_fit
from studio.stages.sync_models import AlignmentMap, Anchor

# Steps of different cameras this close in recorder time are one jump...
MATCH_S = 3.0
# ...when their sizes agree within this (ms, or this fraction of the step).
MATCH_MS = 30.0
MATCH_FRACTION = 0.25
# A camera contradicts a jump when it has confident windows this close on both sides.
CONTRADICT_S = 20.0
CONTRADICT_WINDOWS = 2
# Text words confirm only large jumps: their timing is good to ±100 ms.
TEXT_MIN_MS = 150.0
TEXT_SPAN_S = 60.0
TEXT_MIN_ANCHORS = 4
# Grid of the anchors given to a clip from the jump map.
ANCHOR_STEP_S = 2.0


@dataclass(slots=True)
class RecorderJump:
    recorder_id: str
    rec_s: float
    delta_ms: float  # read the recorder this much later after the jump
    seen_by: list[str] = field(default_factory=list)
    contradicted_by: list[str] = field(default_factory=list)
    text_confirmed_by: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return len(self.seen_by) + len(self.text_confirmed_by) > len(self.contradicted_by)

    def to_dict(self) -> dict[str, Any]:
        return {"recorder": self.recorder_id, "recorder_s": round(self.rec_s, 3),
                "delta_ms": round(self.delta_ms, 1), "seen_by": self.seen_by,
                "text_confirmed_by": self.text_confirmed_by,
                "contradicted_by": self.contradicted_by, "accepted": self.accepted}


def _line(alignment: AlignmentMap) -> tuple[float, float]:
    return alignment.offset, alignment.k or 1.0


def recorder_jumps(
    checks: dict[tuple[str, str], DriftCheck], fits: dict[tuple[str, str], AlignmentMap],
) -> list[RecorderJump]:
    """Jumps of each recorder from the acoustic steps of all its cameras."""
    sightings: dict[str, list[tuple[float, float, str]]] = {}
    for (camera_id, recorder_id), check in checks.items():
        if check.model is None or (camera_id, recorder_id) not in fits:
            continue
        offset, k = _line(check.alignment)
        for step in check.model.steps:
            sightings.setdefault(recorder_id, []).append(
                ((step.t_s - offset) / k, step.delta_ms, camera_id))
    jumps: list[RecorderJump] = []
    for recorder_id, items in sightings.items():
        groups: list[list[tuple[float, float, str]]] = []
        for item in sorted(items):
            for group in groups:
                rec_s = float(np.median([value[0] for value in group]))
                delta = float(np.median([value[1] for value in group]))
                if abs(item[0] - rec_s) <= MATCH_S and abs(item[1] - delta) <= max(
                        MATCH_MS, MATCH_FRACTION * abs(delta)):
                    group.append(item)
                    break
            else:
                groups.append([item])
        for group in groups:
            jumps.append(RecorderJump(
                recorder_id, float(np.median([value[0] for value in group])),
                float(np.median([value[1] for value in group])),
                sorted({value[2] for value in group})))
    for jump in jumps:
        for (camera_id, recorder_id), check in checks.items():
            if recorder_id != jump.recorder_id or camera_id in jump.seen_by or (
                    (camera_id, recorder_id) not in fits):
                continue
            if _contradicts(check, jump):
                jump.contradicted_by.append(camera_id)
        for (camera_id, recorder_id), alignment in fits.items():
            if recorder_id == jump.recorder_id and camera_id not in jump.seen_by and (
                    _text_confirms(alignment, jump)):
                jump.text_confirmed_by.append(camera_id)
    return sorted(jumps, key=lambda item: (item.recorder_id, item.rec_s))


def _contradicts(check: DriftCheck, jump: RecorderJump) -> bool:
    """The camera heard both sides of the jump and found no step there."""
    if check.model is None or not check.confident_s:
        return False
    offset, k = _line(check.alignment)
    rec = (np.asarray(check.confident_s) - offset) / k
    before = int(np.sum((rec < jump.rec_s) & (rec >= jump.rec_s - CONTRADICT_S)))
    after = int(np.sum((rec > jump.rec_s) & (rec <= jump.rec_s + CONTRADICT_S)))
    if before < CONTRADICT_WINDOWS or after < CONTRADICT_WINDOWS:
        return False
    return not any(abs((step.t_s - offset) / k - jump.rec_s) <= MATCH_S
                   for step in check.model.steps)


def _text_residuals(alignment: AlignmentMap) -> list[tuple[float, float]]:
    """``(recorder time, read-later seconds)`` of the word anchors against the line."""
    offset, k = _line(alignment)
    return [(anchor.rec_time, anchor.rec_time - (anchor.cam_time - offset) / k)
            for anchor in alignment.anchors if anchor_regime(anchor) is None]


def _text_confirms(alignment: AlignmentMap, jump: RecorderJump) -> bool:
    if abs(jump.delta_ms) < TEXT_MIN_MS:
        return False
    residuals = _text_residuals(alignment)
    before = [value for rec, value in residuals if jump.rec_s - TEXT_SPAN_S <= rec < jump.rec_s]
    after = [value for rec, value in residuals if jump.rec_s < rec <= jump.rec_s + TEXT_SPAN_S]
    if len(before) < TEXT_MIN_ANCHORS or len(after) < TEXT_MIN_ANCHORS:
        return False
    shift = (float(np.median(after)) - float(np.median(before))) * 1000.0
    return abs(shift - jump.delta_ms) <= max(100.0, 0.35 * abs(jump.delta_ms))


def apply_jumps(
    alignment: AlignmentMap, jumps: list[RecorderJump], clip_duration_s: float,
) -> AlignmentMap | None:
    """``alignment`` (a text map) bent by the accepted jumps inside the clip, or None."""
    offset, k = _line(alignment)
    rec0, rec1 = -offset / k, (clip_duration_s - offset) / k
    inside = sorted((jump for jump in jumps if jump.accepted and rec0 + 1.0 < jump.rec_s
                     < rec1 - 1.0), key=lambda item: item.rec_s)
    if not inside:
        return None

    def shift(rec: float) -> float:
        """Seconds the recorder is read later at ``rec``, relative to the clip start."""
        return sum(jump.delta_ms for jump in inside if jump.rec_s < rec) / 1000.0

    words = [anchor for anchor in alignment.anchors if anchor_regime(anchor) is None]
    if len(words) >= 2:
        # The words lie on one line once every jump is taken out of them.
        straight = [replace(anchor, rec_time=anchor.rec_time - shift(anchor.rec_time))
                    for anchor in words]
        offset, k, inliers = ransac_linear_fit(straight)
        residual = float(np.median([abs(offset + k * a.rec_time - a.cam_time)
                                    for a in inliers])) * 1000.0
    else:
        inliers, residual = [], alignment.residual_ms
    anchors = []
    t = 0.0
    while t <= clip_duration_s + 1e-9:
        rec = (t - offset) / k
        rec += shift(rec + shift(rec))
        regime = sum(1 for jump in inside if jump.rec_s < rec)
        anchors.append(Anchor(t, rec, f"{JUMP_TOKEN}{regime}", 0.5))
        t += ANCHOR_STEP_S
    start = anchors[0].rec_time
    return replace(
        alignment, anchors=anchors, offset=-k * start, k=k, residual_ms=residual,
        provenance=f"{alignment.provenance}+jumps",
        inliers=max(alignment.inliers, len(inliers)),
        notes=[*alignment.notes, "recorder jumps applied at "
               + ", ".join(f"{jump.rec_s:.1f} s ({jump.delta_ms:+.0f} ms)" for jump in inside)],
    )


def unseen_jumps(
    check: DriftCheck, jumps: list[RecorderJump], clip_duration_s: float,
) -> list[RecorderJump]:
    """Accepted jumps inside an acoustically checked clip that its own model lacks."""
    if check.model is None:
        return []
    offset, k = _line(check.alignment)
    rec0, rec1 = -offset / k, (clip_duration_s - offset) / k
    own = [(step.t_s - offset) / k for step in check.model.steps]
    return [jump for jump in jumps
            if jump.accepted and rec0 + 1.0 < jump.rec_s < rec1 - 1.0
            and not any(abs(value - jump.rec_s) <= MATCH_S for value in own)]
