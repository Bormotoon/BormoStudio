from pathlib import Path

from studio.core.recorded_at import recorded_at


def test_quicktime_creationdate_wins_over_the_copy_time() -> None:
    # sony_run DJI_0924: creation_time is when the files were copied.
    tags = {"creation_time": "2026-10-07T20:36:40.000000Z",
            "com.apple.quicktime.creationdate": "2026-10-07T15:47:19Z"}
    result = recorded_at(Path("DJI_0924.mov"), tags, 964.0)
    assert result is not None and result["source"] == "quicktime_creationdate"
    assert result["start"] == "2026-10-07T15:47:19+00:00"
    plain = recorded_at(Path("DJI_0924.mov"), {"creation_time": tags["creation_time"]}, 964.0)
    assert plain is not None and plain["source"] == "creation_time"


def test_a_creationdate_with_a_zone_offset_is_kept_in_utc() -> None:
    result = recorded_at(Path("clip.mov"),
                         {"com.apple.quicktime.creationdate": "2026-10-07T18:47:19+0300"}, 10.0)
    assert result is not None and result["epoch"] == recorded_at(
        Path("clip.mov"), {"com.apple.quicktime.creationdate": "2026-10-07T15:47:19Z"},
        10.0)["epoch"]
