import json
from pathlib import Path

import pytest

from studio.llm.server import LlamaServer


class _Process:
    pid = 123

    def poll(self) -> None:
        return None


def test_server_refuses_to_stop_process_without_matching_owner_marker(tmp_path: Path) -> None:
    executable = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    server = LlamaServer(executable, model, tmp_path)
    server.process = _Process()  # type: ignore[assignment]
    server.marker_path.write_text(
        json.dumps({"pid": 999, "executable": str(executable.resolve())}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unowned"):
        server.stop()


def test_owner_check_requires_pid_and_executable(tmp_path: Path) -> None:
    executable = tmp_path / "llama-server"
    server = LlamaServer(executable, tmp_path / "model", tmp_path)
    server.marker_path.write_text(
        json.dumps({"pid": 123, "executable": str(executable.resolve())}), encoding="utf-8"
    )
    assert server._owns_process(123)
    assert not server._owns_process(456)


def test_server_arguments_follow_tuning_and_supported_flags() -> None:
    from studio.llm.server import resolve_cache_ram_mb, server_arguments

    settings = {"n_gpu_layers": 999, "ctx_size": 8192, "parallel": 1, "flash_attn": "on",
                "cache_type_k": "q8_0", "cache_ram_mb": 2048, "ctx_checkpoints": 4,
                "extra_args": ["--mlock"], "threads": None}
    args = server_arguments(settings, "llama-server", lambda _exe, flag: flag == "--cache-ram")
    assert args == ["--n-gpu-layers", "999", "--ctx-size", "8192", "--parallel", "1",
                    "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-ram", "2048",
                    "--mlock"]
    assert server_arguments({}, "llama-server", lambda *_: True) == []
    auto = {"cache_ram_mb": "auto", "cache_ram_reserve_mb": 10**9, "cache_ram_min_mb": 256}
    assert resolve_cache_ram_mb(auto) == 256
    assert resolve_cache_ram_mb({"cache_ram_mb": None}) is None


def test_server_log_records_command(tmp_path: Path) -> None:
    import sys

    from studio.llm.server import LlamaServer

    log = tmp_path / "llama" / "server.log"
    server = LlamaServer(Path(sys.executable), tmp_path / "model.gguf", tmp_path, 1,
                         arguments=["-c", "print('offloaded 40/40 layers')"], log_file=log)
    # The fake server ignores llama flags: python -m <model> fails, but output is logged.
    server.start()
    assert server.process is not None
    server.process.wait(timeout=10)
    server.stop()
    text = log.read_text()
    assert "--port 1" in text and "=====" in text
