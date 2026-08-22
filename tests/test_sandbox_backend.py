from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
import shlex
import sys

import pytest

import codecraft.sandbox.docker as docker_module
import codecraft.sandbox.factory as factory_module
from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli.bootstrap import build_tool_registry, load_session_config
from codecraft.config import RuntimeSettings
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox import (
    BubblewrapSandboxBackend,
    DockerSandboxBackend,
    DockerSandboxConfig,
    ProcessSandboxBackend,
    CommandPolicy,
    SandboxBackend,
    SandboxBackendError,
    SandboxBackendType,
    SandboxExecutionRequest,
    SandboxExecutionResult,
    SandboxMode,
    SeatbeltSandboxBackend,
    UnavailableSandboxBackend,
    build_sandbox_backend,
)
from codecraft.sandbox._execution import CapturedProcessOutput
from codecraft.schema.tool import ToolCall
from codecraft.schema.session import SessionSource
from codecraft.tool import ToolContext
from codecraft.tool.builtin.system import BashTool


class _CompletedProcess:
    pid = None

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        raise AssertionError("completed process should not be killed")


def _request(tmp_path, **updates) -> SandboxExecutionRequest:
    values = {
        "command": "python --version",
        "cwd": tmp_path,
        "workspace_root": tmp_path,
        "sandbox_mode": SandboxMode.WORKSPACE_WRITE,
        "network_access": False,
        "timeout_seconds": 30,
    }
    values.update(updates)
    return SandboxExecutionRequest(**values)


def _tool_context(tmp_path) -> ToolContext:
    turn = TurnContext(
        session_id="ses_sandbox",
        turn_id="turn_sandbox",
        cwd=tmp_path,
        model="none",
        model_provider="test",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        network_access=False,
        available_tools=[],
        max_tool_calls=1,
        max_tool_output_chars=80_000,
        created_at=datetime.now(UTC),
    )
    return ToolContext(
        context=turn,
        call=ToolCall(
            call_id="call_sandbox",
            name="bash",
            arguments={"command": "python --version"},
        ),
        command_decision=CommandPolicy().classify("python --version"),
    )


def test_docker_command_applies_isolation_and_resource_limits(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    cwd = workspace / "src"
    cwd.mkdir(parents=True)
    monkeypatch.setenv("SAFE_TOKEN", "secret")
    monkeypatch.setenv("UNLISTED_TOKEN", "hidden")
    backend = DockerSandboxBackend(
        DockerSandboxConfig(
            image="codecraft-sandbox:test",
            cpus=1.5,
            memory_mb=768,
            pids_limit=128,
            tmpfs_mb=64,
        )
    )

    command = backend.build_command(
        _request(
            workspace,
            cwd=cwd,
            command="pytest -q",
            env_allowlist=("SAFE_TOKEN",),
        ),
        container_name="codecraft-test",
    )

    assert command[:5] == ["docker", "run", "--rm", "--name", "codecraft-test"]
    assert "--init" in command
    assert command[command.index("--workdir") + 1] == "/workspace/src"
    assert command[command.index("--memory") + 1] == "768m"
    assert command[command.index("--cpus") + 1] == "1.5"
    assert command[command.index("--pids-limit") + 1] == "128"
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in command
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert f"type=bind,source={workspace},target=/workspace" in command
    assert command[command.index("SAFE_TOKEN") - 1] == "--env"
    assert "SAFE_TOKEN=secret" not in command
    assert not any("UNLISTED_TOKEN" in argument for argument in command)
    assert command[-4:] == [
        "codecraft-sandbox:test",
        "/bin/sh",
        "-lc",
        "pytest -q",
    ]


def test_docker_command_uses_read_only_mount_and_rejects_unsafe_inputs(tmp_path):
    backend = DockerSandboxBackend()
    command = backend.build_command(
        _request(tmp_path, sandbox_mode=SandboxMode.READ_ONLY),
        container_name="codecraft-readonly",
    )

    mount = command[command.index("--mount") + 1]
    assert mount.endswith(",readonly")
    with pytest.raises(ValueError, match="not a CLI option"):
        DockerSandboxConfig(image="--privileged")
    with pytest.raises(SandboxBackendError, match="environment variable"):
        backend.build_command(
            _request(tmp_path, env_allowlist=("BAD-NAME",)),
            container_name="codecraft-bad-env",
        )
    with pytest.raises(SandboxBackendError, match="outside workspace root"):
        backend.build_command(
            _request(tmp_path, cwd=tmp_path.parent),
            container_name="codecraft-escaped",
        )
    with pytest.raises(ValueError, match="output limit"):
        _request(tmp_path, max_output_bytes=0)
    with pytest.raises(ValueError, match="timeout"):
        _request(tmp_path, timeout_seconds=0)


def test_docker_command_mounts_single_workspace(tmp_path):
    backend = DockerSandboxBackend()

    command = backend.build_command(
        _request(tmp_path),
        container_name="codecraft-single-workspace",
    )

    assert command.count("--mount") == 1


def test_docker_full_access_still_exposes_only_writable_workspace(tmp_path):
    command = DockerSandboxBackend().build_command(
        _request(tmp_path, sandbox_mode=SandboxMode.DANGER_FULL_ACCESS),
        container_name="codecraft-full-access",
    )

    assert "--read-only" in command
    assert command.count("--mount") == 1
    mount = command[command.index("--mount") + 1]
    assert mount == f"type=bind,source={tmp_path},target=/workspace"
    assert ",readonly" not in mount


def test_docker_backend_executes_without_host_shell(tmp_path, monkeypatch):
    captured = []

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        captured.append((arguments, kwargs))
        return _CompletedProcess(0, b"Python 3.11\n", b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    backend = DockerSandboxBackend(DockerSandboxConfig(image="sandbox:test"))

    result = asyncio.run(backend.execute(_request(tmp_path)))

    assert result.exit_code == 0
    assert result.stdout == b"Python 3.11\n"
    assert result.metadata["backend"] == "docker"
    arguments, kwargs = captured[0]
    assert arguments[0:2] == ("docker", "run")
    assert arguments[-1] == "python --version"
    assert "shell" not in kwargs


def test_docker_backend_reports_missing_executable(tmp_path, monkeypatch):
    async def missing(*arguments, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    backend = DockerSandboxBackend(executable="missing-docker")

    with pytest.raises(SandboxBackendError, match="Docker executable not found"):
        asyncio.run(backend.execute(_request(tmp_path)))


def test_docker_backend_classifies_engine_failure(tmp_path, monkeypatch):
    async def fake_create_subprocess_exec(*arguments, **kwargs):
        return _CompletedProcess(125, b"", b"Unable to find image locally\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    backend = DockerSandboxBackend()

    result = asyncio.run(backend.execute(_request(tmp_path)))

    assert result.backend_error == "Unable to find image locally"


def test_process_backend_is_explicit_and_filters_environment(tmp_path, monkeypatch):
    captured = []

    async def fake_create_subprocess_shell(*arguments, **kwargs):
        captured.append((arguments, kwargs))
        return _CompletedProcess(125, b"", b"command returned 125\n")

    monkeypatch.setattr(
        asyncio, "create_subprocess_shell", fake_create_subprocess_shell
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret")
    monkeypatch.setenv("SAFE_TOKEN", "forwarded")
    backend = ProcessSandboxBackend()

    result = asyncio.run(
        backend.execute(_request(tmp_path, env_allowlist=("SAFE_TOKEN",)))
    )

    assert result.exit_code == 125
    assert result.backend_error is None
    assert result.metadata["isolation"] == "none"
    _, kwargs = captured[0]
    assert kwargs["env"]["SAFE_TOKEN"] == "forwarded"
    assert "DASHSCOPE_API_KEY" not in kwargs["env"]
    assert "codecraft-process-" in kwargs["env"]["HOME"]


def test_process_backend_removes_workspace_and_relative_path_entries(
    tmp_path,
    monkeypatch,
):
    captured = []

    async def fake_create_subprocess_shell(*arguments, **kwargs):
        captured.append((arguments, kwargs))
        return _CompletedProcess(0, b"", b"")

    workspace = tmp_path / "workspace"
    workspace_bin = workspace / "bin"
    external_bin = tmp_path / "external-bin"
    workspace_bin.mkdir(parents=True)
    external_bin.mkdir()
    workspace_link = workspace / "external-link"
    external_link = tmp_path / "workspace-link"
    try:
        workspace_link.symlink_to(external_bin, target_is_directory=True)
        external_link.symlink_to(workspace_bin, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    monkeypatch.setenv(
        "PATH",
        f"{workspace}{os.pathsep}relative"
        f"{os.pathsep}{workspace_bin}"
        f"{os.pathsep}{workspace_link}"
        f"{os.pathsep}{external_link}"
        f"{os.pathsep}{external_bin}"
        f"{os.pathsep}/usr/bin",
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    asyncio.run(ProcessSandboxBackend().execute(_request(workspace)))

    _, kwargs = captured[0]
    assert kwargs["env"]["PATH"] == f"{external_bin}{os.pathsep}/usr/bin"

    asyncio.run(
        ProcessSandboxBackend().execute(
            _request(workspace, allow_workspace_path_entries=True)
        )
    )

    _, approved_kwargs = captured[1]
    assert approved_kwargs["env"]["PATH"] == os.environ["PATH"]


def test_process_backend_bounds_both_output_streams(tmp_path):
    script = "import os; os.write(1, b'o' * 200_000); os.write(2, b'e' * 200_000)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = asyncio.run(
        ProcessSandboxBackend().execute(
            _request(tmp_path, command=command, max_output_bytes=1024)
        )
    )

    assert result.exit_code == 0
    assert result.stdout == b"o" * 1024
    assert result.stderr == b"e" * 1024
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_process_backend_keeps_timed_out_output_bounded(tmp_path):
    script = (
        "import os\nwhile True:\n os.write(1, b'o' * 8192)\n os.write(2, b'e' * 8192)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = asyncio.run(
        ProcessSandboxBackend().execute(
            _request(
                tmp_path,
                command=command,
                timeout_seconds=1,
                max_output_bytes=1024,
            )
        )
    )

    assert result.timed_out is True
    assert len(result.stdout) <= 1024
    assert len(result.stderr) <= 1024
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group assertion")
def test_process_backend_reaps_command_when_caller_is_cancelled(tmp_path):
    async def run_test() -> None:
        pid_path = tmp_path / "command.pid"
        script = (
            "import os, pathlib, time; "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
            "time.sleep(60)"
        )
        command = f"exec {shlex.quote(sys.executable)} -c {shlex.quote(script)}"
        execution = asyncio.create_task(
            ProcessSandboxBackend().execute(_request(tmp_path, command=command))
        )
        for _ in range(100):
            if pid_path.exists():
                break
            await asyncio.sleep(0.01)
        assert pid_path.exists()
        pid = int(pid_path.read_text(encoding="utf-8"))

        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution

        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(run_test())


def test_seatbelt_command_enforces_workspace_write_and_network_policy(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    backend = SeatbeltSandboxBackend()

    command = backend.build_command(
        _request(workspace, command="pytest -q"),
        temp_root=temp_root,
    )

    profile = command[command.index("-p") + 1]
    assert "(deny file-write*)" in profile
    assert '(subpath (param "WRITABLE_ROOT"))' in profile
    assert "(deny network*)" in profile
    assert f"-DWRITABLE_ROOT={workspace}" in command
    assert f"-DTEMP_ROOT={temp_root}" in command
    assert command[-4:] == ["--", "/bin/sh", "-lc", "pytest -q"]


def test_seatbelt_read_only_does_not_allow_workspace_writes(tmp_path):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    command = SeatbeltSandboxBackend().build_command(
        _request(tmp_path, sandbox_mode=SandboxMode.READ_ONLY),
        temp_root=temp_root,
    )

    assert not any(argument.startswith("-DWRITABLE_ROOT=") for argument in command)


def test_bubblewrap_command_uses_os_namespaces_and_bind_mounts(tmp_path):
    workspace = tmp_path / "workspace"
    cwd = workspace / "src"
    cwd.mkdir(parents=True)
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    backend = BubblewrapSandboxBackend(executable="/usr/bin/bwrap")

    command = backend.build_command(
        _request(workspace, cwd=cwd, command="pytest -q"),
        temp_root=temp_root,
    )

    assert command[0] == "/usr/bin/bwrap"
    assert "--die-with-parent" in command
    assert "--unshare-pid" in command
    assert "--unshare-net" in command
    assert command[command.index("--ro-bind") + 1 :][:2] == ["/", "/"]
    workspace_bind = ["--bind", str(workspace), str(workspace)]
    assert any(
        command[index : index + 3] == workspace_bind
        for index in range(len(command) - 2)
    )
    assert command[command.index("--chdir") + 1] == str(cwd)
    assert command[-4:] == ["--", "/bin/sh", "-lc", "pytest -q"]


def test_native_backends_map_full_access_and_enabled_network(tmp_path):
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    request = _request(
        tmp_path,
        sandbox_mode=SandboxMode.DANGER_FULL_ACCESS,
        network_access=True,
    )

    seatbelt_command = SeatbeltSandboxBackend().build_command(
        request,
        temp_root=temp_root,
    )
    seatbelt_profile = seatbelt_command[seatbelt_command.index("-p") + 1]
    assert "(deny file-write*)" not in seatbelt_profile
    assert "(deny network*)" not in seatbelt_profile
    assert not any(
        argument.startswith("-DWRITABLE_ROOT=") for argument in seatbelt_command
    )

    bubblewrap_command = BubblewrapSandboxBackend().build_command(
        request,
        temp_root=temp_root,
    )
    assert "--unshare-net" not in bubblewrap_command
    root_mount = bubblewrap_command.index("--bind")
    assert bubblewrap_command[root_mount : root_mount + 3] == ["--bind", "/", "/"]


def test_auto_backend_prefers_native_os_sandbox(monkeypatch):
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Darwin")
    assert isinstance(
        build_sandbox_backend(SandboxBackendType.AUTO), SeatbeltSandboxBackend
    )

    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.shutil, "which", lambda name: "/usr/bin/bwrap")
    assert isinstance(
        build_sandbox_backend(SandboxBackendType.AUTO), BubblewrapSandboxBackend
    )


def test_auto_backend_fails_closed_when_native_sandbox_is_unavailable(monkeypatch):
    monkeypatch.setattr(factory_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(factory_module.shutil, "which", lambda name: None)
    backend = build_sandbox_backend(SandboxBackendType.AUTO)

    assert isinstance(backend, UnavailableSandboxBackend)
    with pytest.raises(SandboxBackendError, match="bubblewrap is required"):
        asyncio.run(backend.execute(_request(Path.cwd())))


def test_docker_backend_force_removes_timed_out_container(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = -9

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        return FakeProcess()

    async def fake_communicate(process, *, timeout_seconds, max_output_bytes):
        return CapturedProcessOutput(
            stdout=b"partial",
            stderr=b"",
            stdout_truncated=False,
            stderr_truncated=False,
            timed_out=True,
        )

    removed = []

    async def fake_remove(container_name):
        removed.append(container_name)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(docker_module, "communicate", fake_communicate)
    backend = DockerSandboxBackend()
    monkeypatch.setattr(backend, "_force_remove", fake_remove)

    result = asyncio.run(backend.execute(_request(tmp_path, timeout_seconds=1)))

    assert result.timed_out is True
    assert result.stdout == b"partial"
    assert len(removed) == 1
    assert removed[0].startswith("codecraft-")


def test_docker_backend_force_removes_container_when_cancelled(tmp_path, monkeypatch):
    class FakeProcess:
        returncode = None

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        return FakeProcess()

    async def cancelled_communicate(process, **kwargs):
        raise asyncio.CancelledError

    removed = []

    async def fake_remove(container_name):
        removed.append(container_name)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(docker_module, "communicate", cancelled_communicate)
    backend = DockerSandboxBackend()
    monkeypatch.setattr(backend, "_force_remove", fake_remove)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(backend.execute(_request(tmp_path)))

    assert len(removed) == 1
    assert removed[0].startswith("codecraft-")


def test_docker_backend_cleans_up_when_process_start_is_cancelled(
    tmp_path, monkeypatch
):
    async def cancelled_create_subprocess_exec(*arguments, **kwargs):
        raise asyncio.CancelledError

    removed = []

    async def fake_remove(container_name):
        removed.append(container_name)

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", cancelled_create_subprocess_exec
    )
    backend = DockerSandboxBackend()
    monkeypatch.setattr(backend, "_force_remove", fake_remove)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(backend.execute(_request(tmp_path)))

    assert len(removed) == 1
    assert removed[0].startswith("codecraft-")


def test_docker_force_remove_kills_and_reaps_stuck_cleanup(monkeypatch):
    processes = []

    class BlockingCleanupProcess:
        pid = None
        returncode = None

        def __init__(self):
            self.released = asyncio.Event()
            self.killed = False
            self.waits = 0

        async def wait(self):
            self.waits += 1
            await self.released.wait()
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.released.set()

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        process = BlockingCleanupProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(docker_module, "_DOCKER_REMOVE_TIMEOUT_SECONDS", 0.01)

    asyncio.run(DockerSandboxBackend()._force_remove("codecraft-stuck"))

    assert len(processes) == 1
    assert processes[0].killed is True
    assert processes[0].waits == 2


def test_docker_force_remove_finishes_before_propagating_cancellation(monkeypatch):
    async def run_test():
        entered = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()
        backend = DockerSandboxBackend()

        async def fake_remove(container_name):
            entered.set()
            try:
                await release.wait()
            finally:
                completed.set()

        monkeypatch.setattr(backend, "_force_remove", fake_remove)
        removal = asyncio.create_task(
            backend._force_remove_resiliently("codecraft-cancelled")
        )
        await entered.wait()
        removal.cancel()
        await asyncio.sleep(0)

        assert not removal.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await removal
        assert completed.is_set()

    asyncio.run(run_test())


def test_bash_tool_delegates_execution_to_backend(tmp_path):
    class RecordingBackend(SandboxBackend):
        name = "recording"

        def __init__(self):
            self.requests = []

        async def execute(self, request):
            self.requests.append(request)
            return SandboxExecutionResult(
                exit_code=0,
                stdout=b"Python 3.11\n",
                stderr=b"",
                timed_out=False,
                stdout_truncated=True,
                metadata={"backend": self.name},
            )

    backend = RecordingBackend()
    tool = BashTool(sandbox_backend=backend)

    result = asyncio.run(
        tool.arun(
            tool.args_schema.model_validate({"command": "python --version"}),
            _tool_context(tmp_path),
        )
    )

    assert result.success is True
    assert result.content == "Python 3.11\n"
    assert result.data["stdout_truncated"] is True
    assert result.metadata["backend"] == "recording"
    assert backend.requests[0].workspace_root == tmp_path
    assert backend.requests[0].max_output_bytes == 320_000
    assert backend.requests[0].allow_workspace_path_entries is False

    asyncio.run(
        tool.arun(
            tool.args_schema.model_validate({"command": "python --version"}),
            _tool_context(tmp_path).model_copy(update={"approved": True}),
        )
    )
    assert backend.requests[1].allow_workspace_path_entries is True


def test_runtime_settings_parse_docker_backend():
    settings = RuntimeSettings.model_validate(
        {
            "sandbox": {
                "backend": "docker",
                "network_access": False,
                "env_allowlist": ["CI"],
                "docker": {
                    "image": "sandbox:test",
                    "cpus": 2,
                    "memory_mb": 2048,
                },
            }
        }
    )

    assert settings.sandbox.backend == "docker"
    assert settings.sandbox.docker.image == "sandbox:test"
    assert settings.sandbox.docker.cpus == 2
    assert settings.sandbox.docker.memory_mb == 2048
    assert settings.sandbox.env_allowlist == ["CI"]
    with pytest.raises(ValueError, match="environment variable"):
        RuntimeSettings.model_validate({"sandbox": {"env_allowlist": ["BAD-NAME"]}})


def test_session_config_and_tool_registry_preserve_docker_backend(tmp_path):
    config_path = tmp_path / "docker.toml"
    config_path.write_text(
        """
[sandbox]
backend = "docker"
network_access = false
env_allowlist = ["CI"]

[sandbox.docker]
image = "sandbox:test"
cpus = 1.25
memory_mb = 640
""",
        encoding="utf-8",
    )

    config = load_session_config(
        source=SessionSource.TEST,
        provider="mock",
        model="mock-model",
        codecraft_home=tmp_path / ".codecraft",
        config_path=config_path,
        profile=None,
        approval_policy=ApprovalPolicy.NEVER,
        network=None,
    )
    bash = build_tool_registry(config).get("bash")

    assert config.sandbox_backend == "docker"
    assert config.docker_sandbox.image == "sandbox:test"
    assert config.sandbox_env_allowlist == ["CI"]
    assert config.model_dump(mode="json")["sandbox_backend"] == "docker"
    assert isinstance(bash.sandbox_backend, DockerSandboxBackend)
    assert bash.sandbox_backend.config.memory_mb == 640
