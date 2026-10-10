"""Bounded extra attempts shared by all JSON requests in one stage run."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError

from studio.llm.providers import CompletionProvider, parse_json_response


@dataclass
class JsonRequestBudget:
    retries: int = 1
    remaining: int = 5
    attempts: int = 0
    retries_used: int = 0
    backoff_s: float = 1.0
    backoff_total_s: float = 0.0

    def request(
        self,
        provider: CompletionProvider,
        prompt: str,
        validate: Callable[[Any], Any] | None = None,
    ) -> Any:
        for attempt in range(self.retries + 1):
            self.attempts += 1
            try:
                text = (
                    provider.complete(prompt, refresh=True)
                    if attempt
                    else provider.complete(prompt)
                )
                payload = parse_json_response(text)
                return validate(payload) if validate else payload
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt >= self.retries or self.remaining <= 0:
                    raise
                self.remaining -= 1
                self.retries_used += 1
                if isinstance(exc, OSError):
                    delay = min(30.0, self.backoff_s * (2 ** min(attempt, 10)))
                    if isinstance(exc, HTTPError) and exc.headers is not None:
                        try:
                            retry_after = float(exc.headers.get("Retry-After", "0"))
                            if 0 <= retry_after <= 30:
                                delay = max(delay, retry_after)
                        except (ValueError, TypeError):
                            pass
                    if delay > 0:
                        time.sleep(delay)
                        self.backoff_total_s += delay
