from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from codecraft.core.async_utils import finish_task_before_cancelling
from codecraft.sandbox._execution import (
    communicate,
    kill_process_group,
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

_DOCKER_REMOVE_TIMEOUT_SECONDS = 5.0
_DOCKER_PROCESS_GRACE_SECONDS = 1.0


class DockerSandboxConfig(BaseModel):
    """固定镜像及 CPU、内存、进程数、tmpfs 的容器硬资源限制。"""

    image: str = Field(default="codecraft-sandbox:py311", min_length=1)
    cpus: float = Field(default=1.0, gt=0, le=32)
    memory_mb: int = Field(default=1024, ge=64, le=65_536)
    pids_limit: int = Field(default=256, ge=16, le=32_768)
    tmpfs_mb: int = Field(default=256, ge=16, le=8192)

    @field_validator("image")
    @classmethod
    def validate_image(cls, value: str) -> str:
        """拒绝可能被 docker CLI 解释为选项或多个参数的镜像字符串。"""
        if value.startswith("-") or any(character.isspace() for character in value):
            raise ValueError("Docker image must be a reference, not a CLI option")
        return value


class DockerSandboxBackend(SandboxBackend):
    """使用无特权、只读根和资源限制 Docker 容器的隔离后端。"""

    name = SandboxBackendType.DOCKER.value
    isolation = "container"

    def __init__(
        self,
        config: DockerSandboxConfig | None = None,
        *,
        executable: str = "docker",
    ) -> None:
        """设置经校验的容器配置和 docker 可执行文件。"""
        self.config = config or DockerSandboxConfig()
        self.executable = executable

    async def execute(self, request: SandboxExecutionRequest) -> SandboxExecutionResult:
        """运行唯一命名容器，取消/超时时强制删除并归一化 daemon 错误。

        Docker ``run`` 返回 125 表示后端/daemon 启动失败，而非容器内命令
        失败；stderr 会同时放入 backend_error 供 BashTool 区分错误类型。
        """
        container_name = f"codecraft-{uuid4().hex[:16]}"
        command = self.build_command(request, container_name=container_name)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_group_options(),
            )
        except asyncio.CancelledError:
            await self._force_remove_resiliently(container_name)
            raise
        except FileNotFoundError as exc:
            raise SandboxBackendError(
                f"Docker executable not found: {self.executable}"
            ) from exc
        except OSError as exc:
            raise SandboxBackendError(f"could not start Docker sandbox: {exc}") from exc

        try:
            captured = await communicate(
                process,
                timeout_seconds=request.timeout_seconds,
                max_output_bytes=request.max_output_bytes,
            )
        except BaseException:
            await self._force_remove_resiliently(container_name)
            raise
        if captured.timed_out:
            await self._force_remove_resiliently(container_name)
        backend_error = (
            captured.stderr.decode("utf-8", errors="replace").strip()
            if process.returncode == 125 and not captured.timed_out
            else None
        )
        return SandboxExecutionResult(
            exit_code=process.returncode,
            stdout=captured.stdout,
            stderr=captured.stderr,
            timed_out=captured.timed_out,
            stdout_truncated=captured.stdout_truncated,
            stderr_truncated=captured.stderr_truncated,
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
        """构造不拉镜像、只读根、资源封顶、drop capabilities 的 docker argv。

        workspace 映射到固定 ``/workspace``；READ_ONLY 使用 readonly bind，
        其余模式仅让 workspace 可写。环境变量只按名称从宿主传递，不把值
        拼接进命令；包含逗号的宿主路径因 mount grammar 歧义而拒绝。
        """
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
        """启动 ``docker rm --force`` 并在五秒内尽力收口清理进程。"""
        try:
            cleanup = await asyncio.create_subprocess_exec(
                self.executable,
                "rm",
                "--force",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                **process_group_options(),
            )
        except OSError:
            return
        try:
            async with asyncio.timeout(_DOCKER_REMOVE_TIMEOUT_SECONDS):
                await cleanup.wait()
        except TimeoutError:
            await _terminate_cleanup_process(cleanup)
        except asyncio.CancelledError:
            try:
                await _terminate_cleanup_process(cleanup)
            except asyncio.CancelledError:
                pass
            raise

    async def _force_remove_resiliently(self, container_name: str) -> None:
        """让删除任务在调用方取消期间仍优先完成，吞掉非关键清理错误。"""
        cleanup = asyncio.create_task(self._force_remove(container_name))
        try:
            await finish_task_before_cancelling(cleanup)
        except Exception:
            pass


async def _terminate_cleanup_process(process: asyncio.subprocess.Process) -> None:
    """终止卡住的 docker cleanup 进程并短暂等待其回收。"""
    kill_process_group(process)
    waiter = asyncio.create_task(_bounded_process_wait(process))
    await finish_task_before_cancelling(waiter)


async def _bounded_process_wait(process: asyncio.subprocess.Process) -> None:
    """最多等待一秒让已终止 cleanup 进程退出。"""
    try:
        async with asyncio.timeout(_DOCKER_PROCESS_GRACE_SECONDS):
            await process.wait()
    except TimeoutError:
        return


def _workspace_mount(
    request: SandboxExecutionRequest,
) -> tuple[tuple[str, str, str], str]:
    """把宿主 workspace/cwd 映射成容器挂载三元组与 /workspace cwd。"""
    root, resolved_cwd = workspace_path(request)
    access = "ro" if request.sandbox_mode == SandboxMode.READ_ONLY else "rw"
    relative = resolved_cwd.relative_to(root)
    mapped_cwd = "/workspace"
    if relative.parts:
        mapped_cwd = f"{mapped_cwd}/{relative.as_posix()}"
    return (str(root), "/workspace", access), mapped_cwd
