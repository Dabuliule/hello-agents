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
from codecraft.sandbox.policy import SandboxMode


class SeatbeltSandboxBackend(SandboxBackend):
    """macOS process sandbox backed by the built-in Seatbelt runtime."""

    name = SandboxBackendType.SEATBELT.value
    isolation = "os"

    def __init__(self, *, executable: str = "/usr/bin/sandbox-exec") -> None:
        self.executable = executable

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        _, cwd = workspace_path(request)
        with tempfile.TemporaryDirectory(prefix="codecraft-seatbelt-") as temp:
            command = self.build_command(request, temp_root=Path(temp))
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(cwd),
                    env=sandbox_environment(request, Path(temp)),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **process_group_options(),
                )
            except FileNotFoundError as exc:
                raise SandboxBackendError(
                    f"Seatbelt executable not found: {self.executable}"
                ) from exc
            except OSError as exc:
                raise SandboxBackendError(
                    f"could not start Seatbelt sandbox: {exc}"
                ) from exc
            stdout, stderr, timed_out = await communicate(
                process, timeout_seconds=request.timeout_seconds
            )
        return SandboxExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            metadata={
                "backend": self.name,
                "isolation": self.isolation,
                "network_access": request.network_access,
            },
        )

    def build_command(
        self,
        request: SandboxExecutionRequest,
        *,
        temp_root: Path,
    ) -> list[str]:
        root, _ = workspace_path(request)
        policy = ["(version 1)", "(allow default)"]
        definitions = [("TEMP_ROOT", temp_root.resolve())]

        if request.sandbox_mode != SandboxMode.DANGER_FULL_ACCESS:
            policy.append("(deny file-write*)")
            policy.append('(allow file-write* (literal "/dev/null"))')
            policy.append('(allow file-write* (subpath (param "TEMP_ROOT")))')
            if request.sandbox_mode == SandboxMode.WORKSPACE_WRITE:
                definitions.append(("WRITABLE_ROOT", root))
                policy.append('(allow file-write* (subpath (param "WRITABLE_ROOT")))')

        if not request.network_access:
            policy.append("(deny network*)")

        command = [self.executable, "-p", "\n".join(policy)]
        command.extend(f"-D{key}={value}" for key, value in definitions)
        command.extend(["--", "/bin/sh", "-lc", request.command])
        return command
