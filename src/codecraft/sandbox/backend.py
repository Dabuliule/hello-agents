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
    """一次后端执行所需命令、能力、预算和环境策略的不可变快照。

    BashTool 在治理完成后构造本对象；各后端必须把同一份 workspace/cwd、模式、
    网络开关和预算翻译成自身隔离机制，不能重新读取可能已变化的 Runtime 配置。
    ``allow_workspace_path_entries`` 只允许已审批命令使用 workspace 内 PATH 项，
    避免免审批只读命令被仓库中的同名可执行文件替换。
    """

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
    """不假定成功的原始进程输出、终态、截断和后端诊断。

    命令正常启动但返回非零仍是 Result；只有沙箱自身无法启动、验证或建立隔离时
    才抛 ``SandboxBackendError``。BashTool 据此区分 command_failed 与
    sandbox_backend_error，而不会把用户命令失败误报成基础设施故障。
    """

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
    """宿主进程、原生 OS 和容器执行实现共同遵循的异步接口。

    ``name`` 用于配置和审计，``isolation`` 明确声明实际边界；统一接口只保证请求/
    结果协议一致，不代表每个实现都提供同等级隔离，Process 后端会明确报告 none。
    """

    name: str
    isolation: str

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """在声明的隔离边界内执行请求并返回原始结果。"""
        raise NotImplementedError
