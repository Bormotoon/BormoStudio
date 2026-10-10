from pathlib import Path

from studio.core.project import Asset, AssetKind, AssetRole, Project
from studio.core.timeline import SourcePlacement, TimeDomain
from studio.core.transcript import Segment, Transcript, Word
from studio.stages.base import StageContext
from studio.stages.timeline import TimelineStage


def test_disjoint_transcripts_merge_without_overlap_duplicates(tmp_path: Path) -> None:
    from studio.core.project import Asset, AssetKind, AssetRole

    work = tmp_path / "work"
    work.mkdir()
    assets = [Asset(name, Path(f"{name}.wav"), AssetKind.AUDIO, AssetRole.RECORDER)
              for name in ("rec0", "duplicate", "rec1")]
    paths = {}
    for asset in assets:
        path = work / f"{asset.id}.json"
        Transcript(tmp_path / asset.path, "en", 10, [Segment(
            1, 2, (Word(asset.id, 1, 2),),
        )]).save(path)
        paths[asset.id] = path
    project = Project(tmp_path, work, assets=assets, transcripts=paths, placements=[
        SourcePlacement("rec0", 0, 0, 10), SourcePlacement("duplicate", 0, 0, 10),
        SourcePlacement("rec1", 100, 0, 10),
    ])
    result = TimelineStage().run(StageContext(project, {}, work))
    transcript = Transcript.load(result.artifacts[0])
    assert [(word.text, word.start) for word in transcript.words] == [("rec0", 1), ("rec1", 101)]


def test_timeline_stage_maps_words_without_retranscription(tmp_path: Path) -> None:
    work = tmp_path / "_studio"
    source_path = work / "stages" / "transcribe" / "rec.json"
    source = Transcript(
        Path("rec.wav"),
        "en",
        10,
        [Segment(2, 3, (Word("mapped", 2.25, 2.75),))],
    )
    source.save(source_path)
    project = Project(
        tmp_path,
        work,
        assets=[Asset("rec", Path("rec.wav"), AssetKind.AUDIO, AssetRole.RECORDER)],
        placements=[SourcePlacement("rec", -1.0, 0.0, 10.0, 1.01, "text")],
        transcripts={"rec": source_path},
    )
    result = TimelineStage().run(StageContext(project, {}, work))
    timeline = Transcript.load(result.artifacts[0])
    assert timeline.time_domain is TimeDomain.TIMELINE
    assert timeline.words[0].start == 1.272
    assert timeline.words[0].end == 1.777
    assert timeline.map_id == "placement:rec"
