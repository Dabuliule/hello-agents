from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from codecraft.sandbox._execution import (
    communicate,
    process_group_options,
    validated_environment_names,
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


class DockerSandboxConfig(BaseModel):
    image: str = Field(default="codecraft-sandbox:py311", min_length=1)
    cpus: float = Field(default=1.0, gt=0, le=32)
    memory_mb: int = Field(default=1024, ge=64, le=65_536)
    pids_limit: int = Field(default=256, ge=16, le=32_768)
    tmpfs_mb: int = Field(default=256, ge=16, le=8192)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        if value.startswith("-") or any(character.isspace() for character in value):
            raise ValueError("Docker image must be a reference, not a CLI option")
        return value


class DockerSandboxBackend(SandboxBackend):
    name = SandboxBackendType.DOCKER.value
    isolation = "container"

    def __init__(
        self,
        config: DockerSandboxConfig | None = None,
        *,
        executable: str = "docker",
    ) -> None:
        self.config = config or DockerSandboxConfig()
        self.executable = executable

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        container_name = f"codecraft-{uuid4().hex[:16]}"
        command = self.build_command(request, container_name=container_name)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_group_options(),
            )
        except FileNotFoundError as exc:
            raise SandboxBackendError(
                f"Docker executable not found: {self.executable}"
            ) from exc
        except OSError as exc:
            raise SandboxBackendError(f"could not start Docker sandbox: {exc}") from exc

        stdout, stderr, timed_out = await communicate(
            process, timeout_seconds=request.timeout_seconds
        )
        if timed_out:
            await self._force_remove(container_name)
        backend_error = (
            stderr.decode("utf-8", errors="replace").strip()
            if process.returncode == 125 and not timed_out
            else None
        )
        return SandboxExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            backend_error=backend_error,
            metadata={
                "backend": self.name,
                "isolation": self.isolation,
                "container_name": container_name,
                "image": self.config.image,
                "network_access": request.network_access,
            },
        )

    def build_command(
        self,
        request: SandboxExecutionRequest,
        *,
        container_name: str,
    ) -> list[str]:
        mount, container_cwd = _workspace_mount(request)
        command = [
            self.executable,
            "run",
            "--rm",
            "--name",
            container_name,
            "--init",
            "--pull",
            "never",
            "--workdir",
            container_cwd,
            "--memory",
            f"{self.config.memory_mb}m",
            "--cpus",
            str(self.config.cpus),
            "--pids-limit",
            str(self.config.pids_limit),
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.config.tmpfs_mb}m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--env",
            "HOME=/tmp",
            "--env",
            "XDG_CACHE_HOME=/tmp/.cache",
        ]
        if not request.network_access:
            command.extend(["--network", "none"])
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            command.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
        host_path, container_path, access = mount
        if "," in host_path:
            raise SandboxBackendError(
                "workspace path containing a comma cannot be mounted safely"
            )
        mount_spec = f"type=bind,source={host_path},target={container_path}"
        if access == "ro":
            mount_spec += ",readonly"
        command.extend(["--mount", mount_spec])
        for name in validated_environment_names(request.env_allowlist):
            if name in os.environ:
                command.extend(["--env", name])
        command.extend([self.config.image, "/bin/sh", "-lc", request.command])
        return command

    async def _force_remove(self, container_name: str) -> None:
        try:
            cleanup = await asyncio.create_subprocess_exec(
                self.executable,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(cleanup.wait(), timeout=5)
        except (OSError, asyncio.TimeoutError):
            return


def _workspace_mount(
    request: SandboxExecutionRequest,
) -> tuple[tuple[str, str, str], str]:
    root, resolved_cwd = workspace_path(request)
    access = "ro" if request.sandbox_mode == SandboxMode.READ_ONLY else "rw"
    relative = resolved_cwd.relative_to(root)
    mapped_cwd = "/workspace"
    if relative.parts:
        mapped_cwd = f"{mapped_cwd}/{relative.as_posix()}"
    return (str(root), "/workspace", access), mapped_cwd
