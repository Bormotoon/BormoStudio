"""How many CPU processes heavy CPU-bound steps may use."""

from __future__ import annotations

import os
from typing import Any


def cpu_workers(settings: dict[str, Any]) -> int:
    """``compute.cpu_workers``, or all cores but one when it is 0 (auto)."""
    compute = settings.get("compute", {}) if isinstance(settings, dict) else {}
    value = compute.get("cpu_workers", 0) if isinstance(compute, dict) else 0
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return max(1, (os.cpu_count() or 2) - 1)
