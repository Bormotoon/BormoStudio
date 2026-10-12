"""Local OpenAI-compatible llama.cpp provider with deterministic caching."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from studio.core.workspace import published
from studio.llm.json_utils import extract_first_json_value


class CompletionProvider(Protocol):
    def complete(
        self, prompt: str, *, grammar: str | None = None, refresh: bool = False,
        schemas: tuple[dict[str, Any], ...] = (),
    ) -> str: ...


class WatchdogTimeout(TimeoutError):
    """A streamed response produced no token in time."""


@dataclass(frozen=True, slots=True)
class RequestOptions:
    temperature: float = 0.0
    n_predict: int | None = None
    # Watchdog: stream the response and give up on a silent server.
    watchdog: bool = False
    first_token_timeout_s: float = 120.0
    stall_timeout_s: float = 120.0
    max_retries: int = 0
    retry_backoff_s: float = 2.0

    @classmethod
    def from_settings(cls, llm: dict[str, Any], role: str = "") -> RequestOptions:
        role_options = llm.get("role_options", {}).get(role, {})
        n_predict = role_options.get("n_predict", llm.get("n_predict"))
        return cls(
            temperature=float(role_options.get("temperature", llm.get("temperature", 0.0))),
            n_predict=int(n_predict) if n_predict is not None else None,
            watchdog=bool(llm.get("watchdog", True)),
            first_token_timeout_s=float(llm.get("first_token_timeout_s", 120.0)),
            stall_timeout_s=float(llm.get("stall_timeout_s", 120.0)),
            max_retries=int(llm.get("max_retries", 2)),
            retry_backoff_s=float(llm.get("retry_backoff_s", 2.0)),
        )


@dataclass(frozen=True, slots=True)
class LlamaProvider:
    base_url: str
    model: str
    cache_dir: Path | None = None
    model_identity: str = ""
    timeout_s: float = 600.0
    options: RequestOptions = field(default_factory=RequestOptions)

    def complete(
        self, prompt: str, *, grammar: str | None = None, refresh: bool = False,
        schemas: tuple[dict[str, Any], ...] = (),
    ) -> str:
        """``schemas``: JSON schemas the answer is sampled under, strictest first;
        a server that rejects one (HTTP 400 naming the schema) gets the next, then none."""
        cache = self._cache_path(prompt, grammar, schemas[0] if schemas else None)
        if not refresh and cache and cache.is_file():
            text = cache.read_text(encoding="utf-8")
            if text.strip():
                return text
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.options.temperature,
        }
        if self.options.n_predict is not None:
            body["max_tokens"] = self.options.n_predict
        if grammar:
            body["grammar"] = grammar
        remaining = list(schemas)
        if remaining:
            body["response_format"] = _response_format(remaining.pop(0))
        attempt = 0
        while True:
            try:
                text = self._stream(body) if self.options.watchdog else self._request(body)
                break
            except HTTPError as exc:
                if exc.code == 400 and "response_format" in body and _blames_schema(exc):
                    # This build cannot compile the schema: a simpler one, then none.
                    if remaining:
                        body["response_format"] = _response_format(remaining.pop(0))
                    else:
                        del body["response_format"]
                    continue
                # The same request fails the same way on a client error.
                if exc.code < 500 or attempt >= self.options.max_retries:
                    raise
            except (WatchdogTimeout, URLError, ConnectionError, TimeoutError):
                if attempt >= self.options.max_retries:
                    raise
            time.sleep(self.options.retry_backoff_s * 2 ** attempt)
            attempt += 1
        if not isinstance(text, str) or not text.strip():
            raise ValueError("local LLM returned no text content")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with published(cache) as temporary:
                temporary.write_text(text, encoding="utf-8")
        return text

    def _post(self, body: dict[str, Any], timeout: float) -> Any:
        request = Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urlopen(request, timeout=timeout)  # noqa: S310

    def _request(self, body: dict[str, Any]) -> str:
        with self._post(body, self.timeout_s) as response:
            data = json.load(response)
        text: str = data["choices"][0]["message"]["content"]
        return text

    def _stream(self, body: dict[str, Any]) -> str:
        """Read server-sent tokens; a silent server is abandoned, not waited for."""
        options = self.options
        deadline = time.monotonic() + self.timeout_s
        response = self._post({**body, "stream": True},
                              max(options.first_token_timeout_s, options.stall_timeout_s) + 5)
        lines: queue.Queue[bytes | BaseException | None] = queue.Queue()

        def reader() -> None:
            try:
                for line in response:
                    lines.put(line)
                lines.put(None)
            except BaseException as exc:  # handed to the waiting thread
                lines.put(exc)

        threading.Thread(target=reader, daemon=True).start()
        parts: list[str] = []
        try:
            while True:
                limit = options.stall_timeout_s if parts else options.first_token_timeout_s
                wait = min(limit, deadline - time.monotonic())
                if wait <= 0:
                    raise WatchdogTimeout("local LLM exceeded its request timeout")
                try:
                    item = lines.get(timeout=wait)
                except queue.Empty:
                    stage = "stalled" if parts else "sent no first token"
                    raise WatchdogTimeout(f"local LLM {stage} within {limit:g} s") from None
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                line = item.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                choice = json.loads(payload)["choices"][0]
                parts.append((choice.get("delta") or {}).get("content") or "")
        finally:
            _abort(response)
        return "".join(parts)

    def _cache_path(
        self, prompt: str, grammar: str | None, schema: dict[str, Any] | None = None
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        material = json.dumps(
            {
                "version": 2,
                "endpoint": self.base_url.rstrip("/"),
                "model": self.model,
                "model_identity": self.model_identity,
                "prompt": prompt,
                "grammar": grammar,
                **({"schema": schema} if schema else {}),
                "temperature": 0,
                # Only non-default sampling enters the key, so older caches stay valid.
                **({"sampling": [self.options.temperature, self.options.n_predict]}
                   if self.options.temperature or self.options.n_predict is not None else {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode()).hexdigest()
        return self.cache_dir / f"{digest}.txt"


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}


def _blames_schema(error: HTTPError) -> bool:
    """Whether an HTTP 400 body blames the schema or its grammar."""
    try:
        body = error.read().decode("utf-8", "replace").lower()
    except (OSError, ValueError):
        return False
    return any(token in body for token in ("schema", "grammar", "gbnf", "response_format"))


def _abort(response: Any) -> None:
    """Close a response, waking a reader blocked on its socket."""
    import socket

    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    response.close()


def parse_json_response(text: str) -> Any:
    return extract_first_json_value(text)
