from __future__ import annotations

import asyncio
import os

import pytest

from codecraft.sandbox import (
    DockerSandboxBackend,
    DockerSandboxConfig,
    SandboxExecutionRequest,
    SandboxMode,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CODECRAFT_RUN_DOCKER_TESTS") != "1",
        reason="set CODECRAFT_RUN_DOCKER_TESTS=1 to run Docker integration tests",
    ),
]


def test_docker_backend_enforces_runtime_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("CODECRAFT_ALLOWED_ENV", "forwarded")
    monkeypatch.setenv("CODECRAFT_BLOCKED_ENV", "hidden")
    backend = DockerSandboxBackend(DockerSandboxConfig())
    request = SandboxExecutionRequest(
        command=(
            "printf sandbox-ok > sandbox-output.txt; "
            "pwd; "
            "printf uid=; id -u; "
            "printf network=; "
            "if [ -e /sys/class/net/eth0 ]; then echo present; else echo none; fi; "
            "printf allowed=$CODECRAFT_ALLOWED_ENV\\n; "
            "printf blocked=${CODECRAFT_BLOCKED_ENV-unset}\\n; "
            "mount | head -n 1"
        ),
        cwd=tmp_path,
        workspace_root=tmp_path,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        network_access=False,
        timeout_seconds=30,
        env_allowlist=("CODECRAFT_ALLOWED_ENV",),
    )

    result = asyncio.run(backend.execute(request))
    output = result.stdout.decode("utf-8")

    assert result.exit_code == 0, result.stderr.decode("utf-8", errors="replace")
    assert result.backend_error is None
    assert result.metadata["backend"] == "docker"
    assert (tmp_path / "sandbox-output.txt").read_text(encoding="utf-8") == "sandbox-ok"
    assert "/workspace\n" in output
    assert "network=none" in output
    assert "allowed=forwarded" in output
    assert "blocked=unset" in output
    assert " on / type " in output
    assert "(ro," in output
