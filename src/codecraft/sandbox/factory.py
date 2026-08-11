from __future__ import annotations

import platform
import shutil

from codecraft.sandbox.backend import (
    SandboxBackend,
    SandboxBackendError,
    SandboxBackendType,
    SandboxExecutionRequest,
    SandboxExecutionResult,
)
from codecraft.sandbox.bubblewrap import BubblewrapSandboxBackend
from codecraft.sandbox.docker import DockerSandboxBackend, DockerSandboxConfig
from codecraft.sandbox.process import ProcessSandboxBackend
from codecraft.sandbox.seatbelt import SeatbeltSandboxBackend


class UnavailableSandboxBackend(SandboxBackend):
    """保留自动选择失败原因，并在真正执行时 fail-closed 的占位后端。"""

    name = "unavailable"
    isolation = "none"

    def __init__(self, reason: str) -> None:
        """记录平台缺少安全后端的可操作原因。"""
        self.reason = reason

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """始终拒绝执行，要求用户显式安装或选择不隔离的 process。"""
        raise SandboxBackendError(self.reason)


def build_sandbox_backend(
    backend_type: SandboxBackendType,
    docker: DockerSandboxConfig | None = None,
) -> SandboxBackend:
    """按配置和平台构造后端，AUTO 缺少 OS 沙箱时不静默降级 process。

    macOS 自动使用 Seatbelt；Linux 只有检测到 bwrap 才使用 Bubblewrap；其他
    情况返回 UnavailableSandboxBackend。无隔离的 ProcessBackend 必须显式选。
    """
    if backend_type == SandboxBackendType.AUTO:
        system = platform.system()
        if system == "Darwin":
            return SeatbeltSandboxBackend()
        if system == "Linux":
            executable = shutil.which("bwrap")
            if executable:
                return BubblewrapSandboxBackend(executable=executable)
            return UnavailableSandboxBackend(
                "bubblewrap is required for the automatic Linux sandbox; "
                "install bwrap or explicitly configure backend='process'"
            )
        return UnavailableSandboxBackend(
            f"no automatic OS sandbox is available for {system}; "
            "explicitly configure backend='process' to run without isolation"
        )
    if backend_type == SandboxBackendType.SEATBELT:
        return SeatbeltSandboxBackend()
    if backend_type == SandboxBackendType.BUBBLEWRAP:
        return BubblewrapSandboxBackend(executable=shutil.which("bwrap") or "bwrap")
    if backend_type == SandboxBackendType.DOCKER:
        return DockerSandboxBackend(docker)
    return ProcessSandboxBackend()
