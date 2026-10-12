"""Vision-only subprocess executed by the managed module interpreter."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    from studio.modules.models import ModelManager
    from studio.vision.face_track import TrackingSettings, analyze_clip, build_framing_filter

    manager = ModelManager()
    if not manager.installed("yunet"):
        raise ValueError("verified YuNet model is missing; install it through models install")
    settings = TrackingSettings(
        device=request["device"],
        active_speaker=request["active_speaker"] and manager.installed("light-asd"),
        min_face_size=int(request.get("min_face_size", 40)),
        follow=bool(request.get("follow", True)),
        layout=str(request.get("layout", "speaker")),
        switch=str(request.get("switch", "cut")),
    )
    if request.get("mode") == "camera_evidence":
        from studio.vision.camera_evidence import window_evidence

        evidence = window_evidence(
            Path(request["video"]),
            [(float(start), float(end)) for start, end in request["windows"]],
            settings,
        )
        args.result.write_text(json.dumps(evidence), encoding="utf-8")
        return
    plan = analyze_clip(
        Path(request["video"]),
        0,
        request["duration"],
        settings,
        out_w=request["width"],
        out_h=request["height"],
    )
    value = {
        "filter": None
        if plan is None
        else build_framing_filter(plan, out_w=request["width"], out_h=request["height"]),
        "active_speaker_available": settings.active_speaker,
        # Share of samples with a followed face, for the face-ratio filter.
        "face_rate": None if plan is None else plan.rate,
        "plan": None if plan is None else plan.as_dict(),
    }
    args.result.write_text(json.dumps(value), encoding="utf-8")


if __name__ == "__main__":
    main()
