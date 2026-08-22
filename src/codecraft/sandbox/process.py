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
    """显式选择的宿主机进程执行器，不提供 OS 文件系统或网络隔离。

    它仍复用 cwd 校验、最小环境、临时 HOME、输出上限、超时和进程组清理，解决
    凭据误传、输出撑爆内存和取消后遗留子进程等执行卫生问题；但命令仍拥有当前
    用户在宿主机上的权限，``network_access=False`` 在这里也只是上游策略事实，
    不能阻止未知程序直接创建网络连接。因此生产 AUTO 不会静默降级到本后端。
    """

    name = SandboxBackendType.PROCESS.value
    isolation = "none"

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """以净化环境和独立进程组在宿主机执行，不提供 OS 隔离。

        该后端仍验证 cwd、限制时间/输出并重定向临时 HOME，但 metadata 明确
        标记 ``isolation=none``。CommandPolicy/Approval 可以降低误操作概率，不能
        把宿主进程执行变成强隔离；它只应由用户知情地显式配置。
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
