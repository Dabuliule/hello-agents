from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from codecraft.sandbox._execution import (
    communicate,
    process_group_options,
    sandbox_environment,
    workspace_path,
)
from codecraft.sandbox.backend import (
    SandboxBackend,
    SandboxBackendError,
    SandboxBackendType,
    SandboxExecutionRequest,
    SandboxExecutionResult,
)


class ProcessSandboxBackend(SandboxBackend):
    """Explicit host-process execution without an OS isolation boundary."""

    name = SandboxBackendType.PROCESS.value
    isolation = "none"

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """以净化环境和独立进程组在宿主机执行，不提供 OS 隔离。

        该后端仍验证 cwd、限制时间/输出并重定向临时 HOME，但 metadata 明确
        标记 ``isolation=none``；它只应由用户显式配置。
        """
        _, cwd = workspace_path(request)
        with tempfile.TemporaryDirectory(prefix="codecraft-process-") as temp:
            try:
                process = await asyncio.create_subprocess_shell(
                    request.command,
                    cwd=str(cwd),
                    env=sandbox_environment(request, Path(temp)),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **process_group_options(),
                )
            except OSError as exc:
                raise SandboxBackendError(
                    f"could not start host process: {exc}"
                ) from exc
            captured = await communicate(
                process,
                timeout_seconds=request.timeout_seconds,
                max_output_bytes=request.max_output_bytes,
            )
        return SandboxExecutionResult(
            exit_code=process.returncode,
            stdout=captured.stdout,
            stderr=captured.stderr,
            timed_out=captured.timed_out,
            stdout_truncated=captured.stdout_truncated,
            stderr_truncated=captured.stderr_truncated,
            metadata={
                "backend": self.name,
                "isolation": self.isolation,
                "network_access": request.network_access,
            },
        )
