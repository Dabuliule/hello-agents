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
    """通过 macOS ``sandbox-exec``/Seatbelt profile 限制写入和网络的 OS 后端。

    Profile 从 ``allow default`` 开始，再按请求追加 file-write/network deny 与精确
    allow：READ_ONLY 只允许临时目录和 /dev/null 写，WORKSPACE_WRITE 再开放
    workspace，DANGER_FULL_ACCESS 不限制文件写。默认允许宿主文件读取，因此该后端
    主要保护完整性而非隐藏宿主敏感文件；CommandPolicy/Approval 仍不能替代机密隔离。
    """

    name = SandboxBackendType.SEATBELT.value
    isolation = "os"

    def __init__(self, *, executable: str = "/usr/bin/sandbox-exec") -> None:
        """设置 macOS sandbox-exec 路径。"""
        self.executable = executable

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """以生成的 Seatbelt profile 启动 shell 并有界捕获输出。"""
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
        """把 SandboxMode/网络能力翻译成参数化 Seatbelt profile 与 shell argv。

        非 full-access 先拒绝所有 file-write，再仅开放 /dev/null、临时目录，
        WORKSPACE_WRITE 额外开放 workspace；network_access=False 拒绝网络。
        路径通过 ``-D`` 参数传入而不是拼进 profile 语法，避免特殊字符改变规则。
        ``allow default`` 意味着 read-only/workspace-write 仍可读取宿主其他路径。
        """
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
