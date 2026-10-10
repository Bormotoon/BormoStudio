import json
from pathlib import Path

from studio.core.project import Asset, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Transcript
from studio.reels.analysis.contracts import coerce_moment_record
from studio.reels.analysis.replay import annotate_replay, load_heatmap, span_intensity
from studio.reels.analysis.scoring import combined_priority_score


def _info(tmp_path: Path) -> Path:
    path = tmp_path / "source.info.json"
    path.write_text(json.dumps({"heatmap": [
        {"start_time": 0, "end_time": 10, "value": 0.2},
        {"start_time": 10, "end_time": 20, "value": 1.0},
        {"start_time": 20, "end_time": 30, "value": "bad"},
    ]}), encoding="utf-8")
    return path


def test_heatmap_intensity_is_duration_weighted(tmp_path: Path) -> None:
    heatmap = load_heatmap(_info(tmp_path))
    assert len(heatmap) == 2
    assert span_intensity(heatmap, [(10, 20)]) == 1.0
    assert span_intensity(heatmap, [(5, 15)]) == 0.6
    assert span_intensity(heatmap, [(40, 50)]) is None
    assert load_heatmap(tmp_path / "missing.json") == []


def test_replayed_moment_ranks_higher_through_its_placement(tmp_path: Path) -> None:
    media = tmp_path / "source.mp4"
    media.write_bytes(b"x")
    project = Project(tmp_path, tmp_path / "_studio")
    project.assets = [Asset("cam", Path("source.mp4"), "video", AssetRole.CAMERA)]
    project.placements = [SourcePlacement("cam", 5, 0, 30)]
    project.outputs["fetch"] = [media, _info(tmp_path)]
    transcript = Transcript(media, "en", 35, [], time_domain=TimeDomain.TIMELINE)
    records = [coerce_moment_record({"start": start, "end": start + 10, "title": "t",
                                     "quote": "a b", "score": 7})
               for start in (5, 15)]
    annotated, report = annotate_replay(records, project, transcript, project.outputs["fetch"][1])
    assert report["status"] == "applied"
    assert [record.replay_score for record in annotated] == [0.2, 1.0]
    low, high = (combined_priority_score(record.to_dict(), target_min=5, target_max=20)
                 for record in annotated)
    assert high > low
