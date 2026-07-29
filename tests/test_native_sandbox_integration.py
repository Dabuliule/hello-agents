from __future__ import annotations

import asyncio
from dataclasses import replace
import os
import platform
import socket

import pytest

from codecraft.sandbox import (
    BubblewrapSandboxBackend,
    SandboxBackend,
    SandboxExecutionRequest,
    SandboxMode,
    SeatbeltSandboxBackend,
)

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    platform.system() != "Darwin"
    or os.environ.get("CODECRAFT_RUN_NATIVE_SANDBOX_TESTS") != "1",
    reason="set CODECRAFT_RUN_NATIVE_SANDBOX_TESTS=1 on macOS",
)
def test_seatbelt_enforces_write_network_and_environment_boundaries(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-not-leak")
    request = SandboxExecutionRequest(
        command=(
            "printf allowed > inside.txt; "
            f"(printf denied > {outside}) || true; "
            "printf key=${DASHSCOPE_API_KEY-unset}; "
            "python -c 'import socket; socket.socket().bind((\"127.0.0.1\", 0))' "
            ">/dev/null 2>&1 && printf ' network=open' || printf ' network=blocked'"
        ),
        cwd=workspace,
        workspace_root=workspace,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        network_access=False,
        timeout_seconds=30,
    )

    result = asyncio.run(SeatbeltSandboxBackend().execute(request))

    assert result.exit_code == 0, result.stderr.decode("utf-8", errors="replace")
    assert (workspace / "inside.txt").read_text(encoding="utf-8") == "allowed"
    assert not outside.exists()
    assert result.stdout == b"key=unset network=blocked"
    assert result.metadata["isolation"] == "os"
    _assert_read_only_blocks_workspace_write(
        SeatbeltSandboxBackend(),
        request,
        workspace,
    )


@pytest.mark.skipif(
    platform.system() != "Linux"
    or os.environ.get("CODECRAFT_RUN_BUBBLEWRAP_TESTS") != "1",
    reason="set CODECRAFT_RUN_BUBBLEWRAP_TESTS=1 on Linux",
)
def test_bubblewrap_enforces_write_network_and_environment_boundaries(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-not-leak")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener_port = listener.getsockname()[1]
        request = SandboxExecutionRequest(
            command=(
                "printf allowed > inside.txt; "
                f"(printf denied > {outside}) 2>/dev/null || true; "
                "printf key=${DASHSCOPE_API_KEY-unset}; "
                "python -c 'import socket; "
                f'socket.create_connection(("127.0.0.1", {listener_port}), 1)'
                "' >/dev/null 2>&1 "
                "&& printf ' network=open' || printf ' network=blocked'"
            ),
            cwd=workspace,
            workspace_root=workspace,
            sandbox_mode=SandboxMode.WORKSPACE_WRITE,
            network_access=False,
            timeout_seconds=30,
        )

        backend = BubblewrapSandboxBackend()
        result = asyncio.run(backend.execute(request))

    assert result.exit_code == 0, result.stderr.decode("utf-8", errors="replace")
    assert (workspace / "inside.txt").read_text(encoding="utf-8") == "allowed"
    assert not outside.exists()
    assert result.stdout == b"key=unset network=blocked"
    assert result.metadata["isolation"] == "os"
    _assert_read_only_blocks_workspace_write(backend, request, workspace)


def _assert_read_only_blocks_workspace_write(
    backend: SandboxBackend,
    request: SandboxExecutionRequest,
    workspace,
) -> None:
    target = workspace / "read-only-denied.txt"
    read_only_request = replace(
        request,
        command=(
            'if [ "$(cat inside.txt)" != "allowed" ]; then exit 8; fi; '
            "if (printf denied > read-only-denied.txt) 2>/dev/null; then "
            "printf write=open; exit 9; fi; "
            "printf 'read=allowed write=blocked'"
        ),
        sandbox_mode=SandboxMode.READ_ONLY,
    )

    result = asyncio.run(backend.execute(read_only_request))

    assert result.exit_code == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.stdout == b"read=allowed write=blocked"
    assert not target.exists()
