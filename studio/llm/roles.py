"""Per-task model routing preserving managed-server ownership."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from studio.llm.providers import LlamaProvider, RequestOptions
from studio.llm.session import model_identity


def role_provider(
    settings: dict[str, Any],
    section: str,
    role: str,
    cache_dir: Path | None,
    factory: Callable[..., LlamaProvider] = LlamaProvider,
) -> LlamaProvider:
    config = settings.get(section, {})
    override = settings.get("llm", {}).get("roles", {}).get(role, {})
    endpoint = override.get("base_url", config.get("base_url", "http://127.0.0.1:8080"))
    if settings.get("llm", {}).get("managed"):
        endpoint = f"http://127.0.0.1:{settings['llm'].get('port', 8080)}"
    llm = settings.get("llm", {})
    timeout = llm.get("role_options", {}).get(role, {}).get(
        "timeout_s", config.get("request_timeout_s", 600))
    return factory(
        endpoint,
        override.get("model", config.get("model", "local")),
        cache_dir,
        model_identity(settings),
        timeout,
        RequestOptions.from_settings(llm, role),
    )
