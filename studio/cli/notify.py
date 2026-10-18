"""External notifications when an unattended batch or channel run ends.

Targets are deliberately narrow: an HTTP(S) webhook that receives the JSON
summary, and the desktop notifier of the platform invoked with fixed
arguments. Arbitrary shell commands stay disabled (see PROJECT_DECISIONS.md,
"Unattended batch safety"). A failed notification is recorded in the run
report and never changes the run status.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

Opener = Callable[..., Any]
TITLE = "BormoStudio"


@dataclass(frozen=True, slots=True)
class NotifyOptions:
    when: str = "failure"
    webhook: str = ""
    desktop: bool = False
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.when not in {"always", "failure", "never"}:
            raise ValueError("notify when must be always, failure or never")
        if self.webhook and urlsplit(self.webhook).scheme not in {"http", "https"}:
            raise ValueError("notify webhook must be an http or https URL")

    @property
    def configured(self) -> bool:
        return bool(self.webhook or self.desktop)

    def wanted(self, status: str) -> bool:
        if not self.configured or self.when == "never":
            return False
        return self.when == "always" or status != "ok"


def summary(data: dict[str, Any]) -> str:
    projects = data.get("projects", [])
    counts: dict[str, int] = {}
    for item in projects:
        counts[str(item.get("status"))] = counts.get(str(item.get("status")), 0) + 1
    detail = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    failed = [str(item.get("source", "")) for item in projects if item.get("status") == "failed"]
    text = f"{TITLE}: run {data.get('status')} ({detail or 'no projects'})"
    if failed:
        text += "; failed: " + ", ".join(failed[:5]) + (" …" if len(failed) > 5 else "")
    return text


def send(
    options: NotifyOptions,
    data: dict[str, Any],
    report: str | None,
    *,
    opener: Opener | None = None,
    runner: Callable[..., Any] | None = None,
) -> list[dict[str, str]]:
    """Deliver to every configured target; return one outcome per target."""
    opener = opener or urllib.request.urlopen
    runner = runner or subprocess.run
    status = str(data.get("status", "failed"))
    if not options.wanted(status):
        return []
    text = summary(data)
    outcomes: list[dict[str, str]] = []
    if options.webhook:
        payload = json.dumps({
            "text": text, "content": text, "status": status, "report": report,
            "projects": data.get("projects", []),
        }, ensure_ascii=False).encode()
        request = urllib.request.Request(
            options.webhook, data=payload, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "bormostudio"},
        )
        try:
            with opener(request, timeout=options.timeout_s) as response:
                code = getattr(response, "status", 200)
            outcomes.append({"target": "webhook", "status": "sent" if code < 300 else
                             f"http {code}"})
        except OSError as exc:
            outcomes.append({"target": "webhook", "status": "failed", "error": str(exc)})
    if options.desktop:
        command = desktop_command(text)
        if command is None:
            outcomes.append({"target": "desktop", "status": "unavailable"})
        else:
            try:
                result = runner(command, capture_output=True, text=True,
                                timeout=options.timeout_s, check=False)
                outcomes.append({"target": "desktop", "status": "sent" if not result.returncode
                                 else "failed", "error": (result.stderr or "")[-300:]})
            except (OSError, subprocess.SubprocessError) as exc:
                outcomes.append({"target": "desktop", "status": "failed", "error": str(exc)})
    return outcomes


def desktop_command(text: str) -> list[str] | None:
    """Fixed notifier argv; the message is passed as data, never as script."""
    if sys.platform.startswith("linux") and shutil.which("notify-send"):
        return ["notify-send", "--app-name", TITLE, TITLE, text]
    if sys.platform == "darwin" and shutil.which("osascript"):
        return ["osascript", "-e", "on run argv", "-e",
                f'display notification (item 1 of argv) with title "{TITLE}"',
                "-e", "end run", text]
    return None
