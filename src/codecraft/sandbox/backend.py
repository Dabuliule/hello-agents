from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from codecraft.sandbox.policy import SandboxMode


class SandboxBackendType(StrEnum):
    AUTO = "auto"
    PROCESS = "process"
    SEATBELT = "seatbelt"
    BUBBLEWRAP = "bubblewrap"
    DOCKER = "docker"


@dataclass(frozen=True, slots=True)
class SandboxExecutionRequest:
    command: str
    cwd: Path
    workspace_root: Path
    sandbox_mode: SandboxMode
    network_access: bool
    timeout_seconds: int
    env_allowlist: tuple[str, ...] = ()
    allow_workspace_path_entries: bool = False


@dataclass(frozen=True, slots=True)
class SandboxExecutionResult:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool
    backend_error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class SandboxBackendError(RuntimeError):
    pass


class SandboxBackend:
    name: str
    isolation: str

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        raise NotImplementedError
