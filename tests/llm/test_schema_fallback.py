import io
from urllib.error import HTTPError

import pytest

from studio.llm.providers import LlamaProvider


def _rejection(message: str) -> HTTPError:
    return HTTPError("http://x", 400, "Bad Request", {}, io.BytesIO(message.encode()))


def test_a_rejected_schema_steps_down_to_simpler_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[object] = []

    def request(self: LlamaProvider, body: dict) -> str:
        sent.append(body.get("response_format", {}).get("json_schema", {}).get("schema"))
        if len(sent) < 3:
            raise _rejection('{"error": "failed to parse json_schema into a grammar"}')
        return '{"ok": true}'

    monkeypatch.setattr(LlamaProvider, "_request", request)
    full, simple = {"type": "object", "required": ["a"]}, {"type": "object"}
    provider = LlamaProvider("http://127.0.0.1:1", "m")
    assert provider.complete("hi", schemas=(full, simple)) == '{"ok": true}'
    assert sent == [full, simple, None]


def test_other_bad_requests_still_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(self: LlamaProvider, body: dict) -> str:
        raise _rejection('{"error": "context size exceeded"}')

    monkeypatch.setattr(LlamaProvider, "_request", request)
    with pytest.raises(HTTPError):
        LlamaProvider("http://127.0.0.1:1", "m").complete("hi", schemas=({"type": "object"},))
