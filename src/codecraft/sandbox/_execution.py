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
    start_new_session: bool


@dataclass(frozen=True, slots=True)
class CapturedProcessOutput:
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool


@dataclass(slots=True)
class _CaptureBuffer:
    chunks: list[bytes] = field(default_factory=list)
    retained_bytes: int = 0
    truncated: bool = False

    def append(self, chunk: bytes, *, max_bytes: int) -> None:
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
        return b"".join(self.chunks)


async def communicate(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> CapturedProcessOutput:
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
    cleanup = asyncio.create_task(_settle_after_kill(completion, tasks))
    try:
        await finish_task_before_cancelling(cleanup)
    except asyncio.CancelledError:
        pass


async def _settle_after_kill(
    completion: asyncio.Future[tuple[int, None, None]],
    tasks: tuple[asyncio.Task[int], asyncio.Task[None], asyncio.Task[None]],
) -> None:
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
    while chunk := await stream.read(_CAPTURE_CHUNK_BYTES):
        capture.append(chunk, max_bytes=max_bytes)


def kill_process_group(process: asyncio.subprocess.Process) -> None:
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
    return {"start_new_session": True} if os.name != "nt" else {}


def sandbox_environment(
    request: SandboxExecutionRequest,
    temp_root: Path,
) -> dict[str, str]:
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
    """Drop PATH entries that allow a workspace-local executable to shadow tools."""
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
    return candidate == root or root in candidate.parents


def validated_environment_names(names: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [name for name in names if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise SandboxBackendError(f"invalid environment variable names: {invalid}")
    return tuple(dict.fromkeys(names))


def workspace_path(
    request: SandboxExecutionRequest,
) -> tuple[Path, Path]:
    root = request.workspace_root.expanduser().resolve()
    cwd = request.cwd.expanduser().resolve()
    if cwd != root and root not in cwd.parents:
        raise SandboxBackendError("command cwd is outside workspace root")
    return root, cwd
