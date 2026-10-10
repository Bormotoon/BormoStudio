"""Local OpenAI-compatible llama.cpp provider with deterministic caching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from studio.core.workspace import published
from studio.llm.json_utils import extract_first_json_value


class CompletionProvider(Protocol):
    def complete(
        self, prompt: str, *, grammar: str | None = None, refresh: bool = False
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class LlamaProvider:
    base_url: str
    model: str
    cache_dir: Path | None = None
    model_identity: str = ""
    timeout_s: float = 600.0

    def complete(self, prompt: str, *, grammar: str | None = None, refresh: bool = False) -> str:
        cache = self._cache_path(prompt, grammar)
        if not refresh and cache and cache.is_file():
            text = cache.read_text(encoding="utf-8")
            if text.strip():
                return text
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        if grammar:
            body["grammar"] = grammar
        request = Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
            data = json.load(response)
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("local LLM returned no text content")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with published(cache) as temporary:
                temporary.write_text(text, encoding="utf-8")
        return text

    def _cache_path(self, prompt: str, grammar: str | None) -> Path | None:
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
                "temperature": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode()).hexdigest()
        return self.cache_dir / f"{digest}.txt"


def parse_json_response(text: str) -> Any:
    return extract_first_json_value(text)
