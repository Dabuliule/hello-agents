from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

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
from codecraft.schema.tool import ToolCall
from codecraft.schema.session import SessionSource
from codecraft.tool import ToolContext
from codecraft.tool.builtin.system import BashTool


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


def test_docker_command_mounts_single_workspace(tmp_path):
    backend = DockerSandboxBackend()

    command = backend.build_command(
        _request(tmp_path),
        container_name="codecraft-single-workspace",
    )

    assert command.count("--mount") == 1


def test_docker_backend_executes_without_host_shell(tmp_path, monkeypatch):
    captured = []

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            return b"Python 3.11\n", b""

        def kill(self):
            raise AssertionError("successful process should not be killed")

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        captured.append((arguments, kwargs))
        return FakeProcess()

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
    class FakeProcess:
        returncode = 125

        async def communicate(self):
            return b"", b"Unable to find image locally\n"

        def kill(self):
            raise AssertionError("completed process should not be killed")

    async def fake_create_subprocess_exec(*arguments, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    backend = DockerSandboxBackend()

    result = asyncio.run(backend.execute(_request(tmp_path)))

    assert result.backend_error == "Unable to find image locally"


def test_process_backend_is_explicit_and_filters_environment(tmp_path, monkeypatch):
    captured = []

    class FakeProcess:
        returncode = 125

        async def communicate(self):
            return b"", b"command returned 125\n"

        def kill(self):
            raise AssertionError("completed process should not be killed")

    async def fake_create_subprocess_shell(*arguments, **kwargs):
        captured.append((arguments, kwargs))
        return FakeProcess()

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

    async def fake_communicate(process, *, timeout_seconds):
        return b"partial", b"", True

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
    assert result.metadata["backend"] == "recording"
    assert backend.requests[0].workspace_root == tmp_path


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
