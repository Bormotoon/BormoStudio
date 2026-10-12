"""The frozen Forge snapshot, importable for parity tests."""

import sys
import types
from pathlib import Path

import pytest

FORGE = Path(__file__).resolve().parents[2] / "legacy" / "forge"


@pytest.fixture
def forge_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put legacy/forge on the path; skip when it is absent or does not import."""
    if not (FORGE / "podcast_reels_forge").is_dir():
        pytest.skip("the Forge snapshot is not in this checkout")
    for name in ("aiohttp", "requests"):
        # Network clients only; the code under comparison never talks to a server.
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.syspath_prepend(str(FORGE))
    try:
        import podcast_reels_forge.stages.analyze_stage  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"the Forge snapshot does not import here: {exc}")
