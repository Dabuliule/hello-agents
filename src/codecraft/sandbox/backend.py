from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from codecraft.sandbox.policy import SandboxMode


class SandboxBackendType(StrEnum):
    """可配置的自动、宿主进程、OS 与容器执行后端。"""

    AUTO = "auto"
    PROCESS = "process"
    SEATBELT = "seatbelt"
    BUBBLEWRAP = "bubblewrap"
    DOCKER = "docker"


@dataclass(frozen=True, slots=True)
class SandboxExecutionRequest:
    """后端执行所需命令、路径、安全模式、预算和环境白名单快照。"""

    command: str
    cwd: Path
    workspace_root: Path
    sandbox_mode: SandboxMode
    network_access: bool
    timeout_seconds: int
    max_output_bytes: int = 320_000
    env_allowlist: tuple[str, ...] = ()
    allow_workspace_path_entries: bool = False

    def __post_init__(self) -> None:
        """拒绝会让超时或捕获语义失效的非正预算。"""
        if self.timeout_seconds <= 0:
            raise ValueError("sandbox timeout must be positive")
        if self.max_output_bytes <= 0:
            raise ValueError("sandbox output limit must be positive")


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    """不假定成功的原始进程输出、终态、截断和后端诊断。"""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    backend_error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SandboxBackendError(RuntimeError):
    """沙箱本身无法启动或验证，而非被执行命令返回非零。"""


class SandboxBackend:
    """所有命令隔离实现必须遵循的异步后端接口。"""

    name: str
    isolation: str

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """在声明的隔离边界内执行请求并返回原始结果。"""
        raise NotImplementedError
