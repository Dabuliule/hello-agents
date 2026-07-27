from __future__ import annotations

import asyncio
import os
import re
import signal
from pathlib import Path
from typing import TypedDict

from codecraft.sandbox.backend import (
    SandboxBackendError,
    SandboxExecutionRequest,
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
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


async def communicate(
    process: asyncio.subprocess.Process,
    *,
    timeout_seconds: int,
) -> tuple[bytes, bytes, bool]:
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
        return stdout, stderr, False
    except asyncio.TimeoutError:
        kill_process_group(process)
        stdout, stderr = await process.communicate()
        return stdout, stderr, True


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
    process.kill()


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


def validated_environment_names(names: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [name for name in names if not _ENV_NAME.fullmatch(name)]
    if invalid:
        raise SandboxBackendError(f"invalid environment variable names: {invalid}")
    return tuple(dict.fromkeys(names))


def workspace_paths(
    request: SandboxExecutionRequest,
) -> tuple[tuple[Path, ...], Path]:
    roots = tuple(
        dict.fromkeys(root.expanduser().resolve() for root in request.workspace_roots)
    )
    if not roots:
        raise SandboxBackendError("sandbox requires a workspace root")
    cwd = request.cwd.expanduser().resolve()
    if not any(cwd == root or root in cwd.parents for root in roots):
        raise SandboxBackendError("command cwd is outside workspace roots")
    return roots, cwd
