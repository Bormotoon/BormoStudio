import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from studio.llm.providers import LlamaProvider, RequestOptions, WatchdogTimeout
from studio.llm.roles import role_provider


class Handler(BaseHTTPRequestHandler):
    plan: list[str] = []
    bodies: list[dict] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        behaviour = type(self).plan.pop(0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if behaviour == "silent":
            time.sleep(1.5)
            return
        for token in ("Hel", "lo"):
            chunk = {"choices": [{"delta": {"content": token}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            if behaviour == "stall":
                time.sleep(1.5)
                return
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server():  # type: ignore[no-untyped-def]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Handler.plan, Handler.bodies = [], []
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _provider(url: str, **options: object) -> LlamaProvider:
    values = {"watchdog": True, "first_token_timeout_s": 0.4, "stall_timeout_s": 0.4,
              "max_retries": 1, "retry_backoff_s": 0.0, **options}
    return LlamaProvider(url, "local", timeout_s=10, options=RequestOptions(**values))  # type: ignore[arg-type]


def test_streamed_tokens_are_joined_with_sampling_options(server: str) -> None:
    Handler.plan = ["ok"]
    text = _provider(server, temperature=0.3, n_predict=64).complete("prompt")
    assert text == "Hello"
    assert Handler.bodies[0]["stream"] is True
    assert (Handler.bodies[0]["temperature"], Handler.bodies[0]["max_tokens"]) == (0.3, 64)


@pytest.mark.parametrize("failure", ["stall", "silent"])
def test_stalled_stream_is_retried(server: str, failure: str) -> None:
    Handler.plan = [failure, "ok"]
    started = time.monotonic()
    assert _provider(server).complete("prompt") == "Hello"
    assert time.monotonic() - started < 1.4  # abandoned, not waited out
    Handler.plan = [failure, failure]
    with pytest.raises(WatchdogTimeout):
        _provider(server).complete("prompt")


def test_role_options_reach_the_provider(tmp_path: Path) -> None:
    settings = {"text": {"base_url": "http://x", "request_timeout_s": 50},
                "llm": {"temperature": 0.1, "max_retries": 3,
                        "role_options": {"article": {"temperature": 0.4, "timeout_s": 9}}}}
    article = role_provider(settings, "text", "article", tmp_path)
    proofread = role_provider(settings, "text", "proofread", tmp_path)
    assert (article.options.temperature, article.timeout_s) == (0.4, 9)
    assert (proofread.options.temperature, proofread.timeout_s) == (0.1, 50)
    assert proofread.options.watchdog and proofread.options.max_retries == 3
    # Sampling changes the cache key; default sampling keeps the old one.
    assert article._cache_path("p", None) != proofread._cache_path("p", None)
    assert (LlamaProvider("http://x", "local", tmp_path)._cache_path("p", None)
            == LlamaProvider("http://x", "local", tmp_path,
                             options=RequestOptions(watchdog=True))._cache_path("p", None))
