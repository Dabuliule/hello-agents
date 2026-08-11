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


class BubblewrapSandboxBackend(SandboxBackend):
    """Linux process sandbox backed by bubblewrap namespaces and bind mounts."""

    name = SandboxBackendType.BUBBLEWRAP.value
    isolation = "os"

    def __init__(self, *, executable: str = "bwrap") -> None:
        """设置 Bubblewrap 可执行文件名或绝对路径。"""
        self.executable = executable

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """构造 namespace/bind 策略，启动并有界捕获 Linux 沙箱进程。"""
        with tempfile.TemporaryDirectory(prefix="codecraft-bwrap-") as temp:
            command = self.build_command(request, temp_root=Path(temp))
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    env=sandbox_environment(request, Path(temp)),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **process_group_options(),
                )
            except FileNotFoundError as exc:
                raise SandboxBackendError(
                    f"bubblewrap executable not found: {self.executable}"
                ) from exc
            except OSError as exc:
                raise SandboxBackendError(
                    f"could not start bubblewrap sandbox: {exc}"
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

    def build_command(
        self,
        request: SandboxExecutionRequest,
        *,
        temp_root: Path,
    ) -> list[str]:
        """将 SandboxMode 映射为根只读/可写与 workspace 可写挂载。

        READ_ONLY 和 WORKSPACE_WRITE 都先把宿主根只读绑定；后者只覆盖绑定
        workspace 为可写。DANGER_FULL_ACCESS 才把整个根可写绑定。网络关闭
        时创建独立 network namespace。
        """
        root, cwd = workspace_path(request)
        root_mount = (
            "--bind"
            if request.sandbox_mode == SandboxMode.DANGER_FULL_ACCESS
            else "--ro-bind"
        )
        command = [
            self.executable,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
        ]
        if not request.network_access:
            command.append("--unshare-net")
        command.extend(
            [
                root_mount,
                "/",
                "/",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--bind",
                str(temp_root.resolve()),
                str(temp_root.resolve()),
            ]
        )
        if request.sandbox_mode == SandboxMode.WORKSPACE_WRITE:
            command.extend(["--bind", str(root), str(root)])
        command.extend(["--chdir", str(cwd), "--", "/bin/sh", "-lc", request.command])
        return command
