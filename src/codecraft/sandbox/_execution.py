from __future__ import annotations

import asyncio
import os
import re
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from codecraft.core.async_utils import finish_task_before_cancelling
from codecraft.sandbox.backend import (
    SandboxBackendError,
    SandboxExecutionRequest,
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CAPTURE_CHUNK_BYTES = 64 * 1024
_PROCESS_CLEANUP_SECONDS = 5.0
_SAFE_ENV_NAMES = frozenset(
    {
        "COLORTERM",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "PATH",
        "PATHEXT",
        "SHELL",
        "SYSTEMROOT",
        "TERM",
        "USER",
        "WINDIR",
    }
)


class ProcessGroupOptions(TypedDict, total=False):
    """跨平台传给 asyncio 子进程创建函数的进程组参数。"""

    start_new_session: bool


@dataclass(frozen=True, slots=True)
class CapturedProcessOutput:
    """有界捕获的 stdout/stderr 及各自截断与整体超时事实。"""

    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool


@dataclass(slots=True)
class _CaptureBuffer:
    """持续排空管道但只保留前 max_bytes，避免子进程因反压死锁。"""

    chunks: list[bytes] = field(default_factory=list)
    retained_bytes: int = 0
    truncated: bool = False

    def append(self, chunk: bytes, *, max_bytes: int) -> None:
        """追加可保留前缀，超出的数据只标记 truncated。"""
        remaining = max_bytes - self.retained_bytes
        if remaining <= 0:
            self.truncated = True
            return
        retained = chunk[:remaining]
        self.chunks.append(retained)
        self.retained_bytes += len(retained)
        if len(retained) < len(chunk):
            self.truncated = True

    def value(self) -> bytes:
        """合并已保留字节块。"""
        return b"".join(self.chunks)


async def communicate(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> CapturedProcessOutput:
    """并发等待进程并排空两条管道，超时/取消时杀死整个进程组。

    Args:
        process: stdout、stderr 均配置为 PIPE 的已启动子进程。
        timeout_seconds: 进程等待与排空共用的正数上限。
        max_output_bytes: stdout 与 stderr 各自最多保留的正数字节数。

    Returns:
        即使超限仍已把管道排空的有界输出；超时由 timed_out 表示。

    Cancellation:
        先 kill 进程组并等待清理任务收口，再重新抛出 CancelledError，避免
        shell 启动的孙进程遗留。
    """
    stdout_stream, stderr_stream = _validated_capture_streams(
        process,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )

    stdout = _CaptureBuffer()
    stderr = _CaptureBuffer()
    wait_task = asyncio.create_task(process.wait())
    stdout_task = asyncio.create_task(
        _drain_stream(stdout_stream, stdout, max_bytes=max_output_bytes)
    )
    stderr_task = asyncio.create_task(
        _drain_stream(stderr_stream, stderr, max_bytes=max_output_bytes)
    )
    tasks = (wait_task, stdout_task, stderr_task)
    completion = asyncio.gather(*tasks)

    timed_out = False
    try:
        async with asyncio.timeout(timeout_seconds):
            await asyncio.shield(completion)
    except TimeoutError:
        timed_out = True
        kill_process_group(process)
        await _settle_after_kill(completion, tasks)
    except asyncio.CancelledError:
        kill_process_group(process)
        await _settle_before_cancelling(completion, tasks)
        raise
    except Exception:
        kill_process_group(process)
        await _settle_after_kill(completion, tasks)
        raise

    return CapturedProcessOutput(
        stdout=stdout.value(),
        stderr=stderr.value(),
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
        timed_out=timed_out,
    )


def _validated_capture_streams(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamReader]:
    """校验正预算和 PIPE 配置后返回两条捕获流。"""
    if timeout_seconds <= 0:
        raise ValueError("process timeout must be positive")
    if max_output_bytes <= 0:
        raise ValueError("process output limit must be positive")
    if process.stdout is None or process.stderr is None:
        raise ValueError("process stdout and stderr must both be pipes")
    return process.stdout, process.stderr


async def _settle_before_cancelling(
    completion: asyncio.Future[tuple[int, None, None]],
    tasks: tuple[asyncio.Task[int], asyncio.Task[None], asyncio.Task[None]],
) -> None:
    """在传播外层取消前，用不可丢失清理任务等待 kill 收口。"""
    cleanup = asyncio.create_task(_settle_after_kill(completion, tasks))
    try:
        await finish_task_before_cancelling(cleanup)
    except asyncio.CancelledError:
        pass


async def _settle_after_kill(
    completion: asyncio.Future[tuple[int, None, None]],
    tasks: tuple[asyncio.Task[int], asyncio.Task[None], asyncio.Task[None]],
) -> None:
    """kill 后最多等待五秒，再取消并回收所有 wait/drain task。"""
    try:
        async with asyncio.timeout(_PROCESS_CLEANUP_SECONDS):
            await asyncio.shield(completion)
    except Exception:
        pass
    finally:
        if not completion.done():
            completion.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _drain_stream(
    stream: asyncio.StreamReader,
    capture: _CaptureBuffer,
    *,
    max_bytes: int,
) -> None:
    """持续读取固定大小块直至 EOF；CaptureBuffer 决定保留量。"""
    while chunk := await stream.read(_CAPTURE_CHUNK_BYTES):
        capture.append(chunk, max_bytes=max_bytes)


def kill_process_group(process: asyncio.subprocess.Process) -> None:
    """POSIX 优先 SIGKILL 独立进程组，失败或 Windows 则 kill 单进程。"""
    pid = getattr(process, "pid", None)
    if os.name != "nt" and isinstance(pid, int):
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            if process.returncode is not None:
                return
        except OSError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        return


def process_group_options() -> ProcessGroupOptions:
    """POSIX 请求新 session，使后续 killpg 能覆盖 shell 的子孙进程。"""
    return {"start_new_session": True} if os.name != "nt" else {}


def sandbox_environment(
    request: SandboxExecutionRequest,
    temp_root: Path,
) -> dict[str, str]:
    """从最小安全集合与显式白名单构造沙箱环境并隔离 HOME/cache。

    未审批命令会从 PATH 移除 workspace 内、相对和无效条目，防止仓库中的
    假 ``git``/``python`` 覆盖系统工具；临时 HOME/TMP/cache 可在只读根下写。
    这只减少环境泄密和命令 shadowing，不限制进程读取绝对路径或访问网络，后者
    必须由 Seatbelt/Bubblewrap/Docker 等实际隔离后端完成。
    """
    names = _SAFE_ENV_NAMES | frozenset(
        validated_environment_names(request.env_allowlist)
    )
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.setdefault("PATH", os.defpath)
    if not request.allow_workspace_path_entries:
        environment["PATH"] = _sandbox_path(
            environment["PATH"],
            workspace_root=request.workspace_root,
        )
    cache_root = temp_root / ".cache"
    cache_root.mkdir(exist_ok=True)
    environment.update(
        {
            "HOME": str(temp_root),
            "TMPDIR": str(temp_root),
            "XDG_CACHE_HOME": str(cache_root),
        }
    )
    return environment


def _sandbox_path(value: str, *, workspace_root: Path) -> str:
    """移除可让 workspace 本地程序覆盖系统工具的 PATH 项。

    同时比较词法路径与解析 symlink 后的真实路径：前者拒绝“路径位于 workspace、
    symlink 目标在外部”，后者拒绝“外部 PATH symlink 反向指入 workspace”。相对/
    空项也丢弃，因为它们会随 cwd 改变解析目标。
    """
    lexical_root = Path(os.path.abspath(workspace_root.expanduser()))
    resolved_root = workspace_root.expanduser().resolve(strict=False)
    selected: list[str] = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            continue
        try:
            lexical_candidate = Path(os.path.abspath(candidate))
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if _is_within(lexical_candidate, lexical_root) or _is_within(
            resolved, resolved_root
        ):
            continue
        selected.append(entry)
    return os.pathsep.join(dict.fromkeys(selected))


def _is_within(candidate: Path, root: Path) -> bool:
    """判断候选路径是否等于根或为其后代。"""
    return candidate == root or root in candidate.parents


def validated_environment_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """验证 shell 环境变量名并按首次出现顺序去重。

    Raises:
        SandboxBackendError: 名称不符合 ``[A-Za-z_][A-Za-z0-9_]*``。
    """
    invalid = [name for name in names if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise SandboxBackendError(f"invalid environment variable names: {invalid}")
    return tuple(dict.fromkeys(names))


def workspace_path(
    request: SandboxExecutionRequest,
) -> tuple[Path, Path]:
    """解析 workspace/cwd 并拒绝 cwd 逃逸，返回二者真实路径。"""
    root = request.workspace_root.expanduser().resolve()
    cwd = request.cwd.expanduser().resolve()
    if cwd != root and root not in cwd.parents:
        raise SandboxBackendError("command cwd is outside workspace root")
    return root, cwd
