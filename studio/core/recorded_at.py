"""When a file started recording, from whatever the device left behind.

Sources, best first: the QuickTime ``com.apple.quicktime.creationdate`` (the
recording time; DJI and Canon keep it when files are copied), the container's
``creation_time`` (QuickTime/DJI, UTC; often the time of copying or ingest:
on sony_run it was hours after the recording and its deltas spread over
10 000 s),
a Broadcast WAV ``bext`` chunk (origination date/time and the sample count
since midnight), a timestamp in the file name (``2026-10-05_103823465``,
``20261005_103823``) and finally ``mtime − duration``. Clocks of different
devices disagree (time zones, unset clocks), so these values are only used
relative to each other within one device: ``locate`` learns the offset
between a camera group's clock and a recorder's from one acoustic match.
Wall-clock values without a zone are stored as if they were UTC.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_NAME_PATTERNS = (
    re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})[_ T-](\d{2})[-:.]?(\d{2})[-:.]?(\d{2})(\d{3})?"),
    re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(\d{3})?(?!\d)"),
)


def recorded_at(path: Path, tags: dict[str, str], duration_s: float,
                sample_rate: int | None = None) -> dict[str, Any] | None:
    """``{"start": ISO, "epoch": seconds, "source": ..., "precision": seconds}``."""
    recorded = _iso(tags.get("com.apple.quicktime.creationdate", ""))
    if recorded is not None:
        return _result(recorded, "quicktime_creationdate", 1.0)
    created = _iso(tags.get("creation_time", ""))
    if created is not None:
        return _result(created, "creation_time", 1.0)
    bwf = _bwf(tags, sample_rate)
    if bwf is not None:
        return _result(bwf, "bwf", 0.001)
    for pattern in _NAME_PATTERNS:
        match = pattern.search(path.stem)
        if match:
            year, month, day, hour, minute, second, millis = match.groups()
            try:
                moment = datetime(int(year), int(month), int(day), int(hour), int(minute),
                                  int(second), int(millis or 0) * 1000, tzinfo=timezone.utc)
            except ValueError:
                continue
            return _result(moment, "file_name", 0.001 if millis else 1.0)
    try:
        modified = path.stat().st_mtime
    except OSError:
        return None
    moment = datetime.fromtimestamp(modified, tz=timezone.utc) - timedelta(seconds=duration_s)
    return _result(moment, "mtime", 2.0)


def epoch(info: dict[str, Any]) -> float | None:
    value = info.get("recorded_at") if isinstance(info, dict) else None
    if isinstance(value, dict) and isinstance(value.get("epoch"), (int, float)):
        return float(value["epoch"])
    return None


def _result(moment: datetime, source: str, precision: float) -> dict[str, Any]:
    return {"start": moment.isoformat(), "epoch": moment.timestamp(), "source": source,
            "precision": precision}


def _iso(value: str) -> datetime | None:
    if not value or "T" not in value and " " not in value.strip():
        return None
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.year < 1990:  # unset camera clocks report 1904/1970
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _bwf(tags: dict[str, str], sample_rate: int | None) -> datetime | None:
    date = tags.get("date") or tags.get("origination_date") or ""
    match = re.fullmatch(r"(\d{4})[-:/](\d{2})[-:/](\d{2})", date.strip())
    if not match:
        return None
    day = datetime(int(match[1]), int(match[2]), int(match[3]), tzinfo=timezone.utc)
    reference = tags.get("time_reference")
    if reference and sample_rate:
        try:
            return day + timedelta(seconds=int(reference) / sample_rate)
        except ValueError:
            pass
    clock = tags.get("origination_time") or tags.get("creation_time") or ""
    time_match = re.fullmatch(r"(\d{2})[-:.](\d{2})[-:.](\d{2})", clock.strip())
    if not time_match:
        return None
    return day + timedelta(hours=int(time_match[1]), minutes=int(time_match[2]),
                           seconds=int(time_match[3]))
