from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp import types
from pydantic import BaseModel, ValidationError

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.config import RuntimeSettings
from codecraft.cli.bootstrap import build_tool_registry, load_session_config
from codecraft.core.errors import ToolNotFoundError
from codecraft.core.turn_context import TurnContext
from codecraft.mcp.client import (
    MCPConnectionError,
    MCPStdioProvider,
    MCPTool,
    mcp_args_model,
    mcp_tool_name,
)
from codecraft.mcp.config import MCPServerSettings, MCPToolPolicySettings
from codecraft.sandbox import SandboxMode
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.session import SessionSource
from codecraft.schema.tool import ToolCall, ToolEffect, ToolResult
from codecraft.tool import AsyncToolProvider, BaseTool, ToolContext, ToolRegistry
from codecraft.tool.runner import ToolRunner


def _turn_context(tmp_path, *, sandbox_mode=SandboxMode.READ_ONLY) -> TurnContext:
    return TurnContext(
        session_id="ses_mcp",
        turn_id="turn_mcp",
        cwd=tmp_path,
        model="none",
        model_provider="test",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=sandbox_mode,
        network_access=False,
        available_tools=[],
        max_tool_calls=1,
        max_tool_output_chars=80_000,
        created_at=datetime.now(UTC),
    )


def test_mcp_settings_parse_conservative_defaults_and_tool_override():
    settings = RuntimeSettings.model_validate(
        {
            "mcp": {
                "servers": {
                    "calculator": {
                        "command": "python",
                        "args": ["server.py"],
                        "env_allowlist": ["API_TOKEN", "API_TOKEN"],
                        "tools": {
                            "add": {
                                "effects": ["read_only"],
                                "requires_approval": False,
                            }
                        },
                    }
                }
            }
        }
    )

    server = settings.mcp.servers["calculator"]
    assert server.default_effects == {"network", "external"}
    assert server.requires_approval is True
    assert server.env_allowlist == ["API_TOKEN"]
    assert server.max_pages == 32
    assert server.max_discovery_bytes == 1_000_000
    assert server.policy_for("add").effects == {"read_only"}
    assert server.policy_for("unknown").requires_approval is True

    with pytest.raises(ValueError, match="server names"):
        RuntimeSettings.model_validate(
            {"mcp": {"servers": {"invalid server": {"command": "python"}}}}
        )


def test_session_config_builds_namespaced_mcp_provider(tmp_path):
    config_path = tmp_path / "mcp.toml"
    config_path.write_text(
        """
[mcp.servers.calculator]
command = "python"
args = ["server.py"]

[mcp.servers.calculator.tools.add]
effects = ["read_only"]
requires_approval = false
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
    registry = build_tool_registry(config)

    assert config.mcp_servers["calculator"].tools["add"].effects == {"read_only"}
    assert config.model_dump(mode="json")["mcp_servers"]["calculator"]["command"] == (
        "python"
    )
    assert registry.async_provider_names() == ["mcp:calculator"]


def test_mcp_argument_model_preserves_and_enforces_remote_json_schema():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer", "minimum": 1}},
        "required": ["count"],
        "additionalProperties": False,
    }
    model = mcp_args_model("count", schema)

    assert model.model_json_schema() == schema
    assert model.model_validate({"count": 2}).model_dump() == {"count": 2}
    with pytest.raises(ValidationError, match="less than the minimum of 1"):
        model.model_validate({"count": 0})
    with pytest.raises(ValidationError, match="required property"):
        model.model_validate({})


def test_mcp_tool_names_are_namespaced_provider_safe_and_bounded():
    assert mcp_tool_name("demo", "hello world") == "mcp__demo__hello_world"

    long_name = mcp_tool_name("demo", "tool/" + "x" * 100)

    assert len(long_name) == 64
    assert long_name.startswith("mcp__demo__tool_")
    assert "/" not in long_name


def test_async_tool_registry_starts_once_and_removes_provider_tools():
    class EchoArgs(BaseModel):
        text: str

    class EchoTool(BaseTool):
        name = "async_echo"
        description = "Echo text."
        args_schema = EchoArgs
        effects = {ToolEffect.READ_ONLY}

        async def arun(self, args: EchoArgs, context: ToolContext) -> ToolResult:
            return ToolResult(success=True, content=args.text)

    class RecordingProvider(AsyncToolProvider):
        name = "recording"

        def __init__(self):
            self.starts = 0
            self.closes = 0

        async def start(self):
            self.starts += 1
            return [EchoTool()]

        async def close(self):
            self.closes += 1

    async def run_test():
        provider = RecordingProvider()
        registry = ToolRegistry(async_providers=[provider])

        await registry.start()
        await registry.start()
        assert registry.get("async_echo").name == "async_echo"
        assert provider.starts == 1

        await registry.close()
        assert provider.closes == 1
        with pytest.raises(ToolNotFoundError):
            registry.get("async_echo")

    asyncio.run(run_test())


def test_async_tool_registry_rolls_back_cancelled_provider_start():
    class RecordingProvider(AsyncToolProvider):
        def __init__(self, name: str, *, block: bool = False) -> None:
            self.name = name
            self.block = block
            self.entered = asyncio.Event()
            self.closes = 0

        async def start(self):
            self.entered.set()
            if self.block:
                await asyncio.Event().wait()
            return []

        async def close(self):
            self.closes += 1

    async def run_test() -> None:
        first = RecordingProvider("first")
        blocked = RecordingProvider("blocked", block=True)
        registry = ToolRegistry(async_providers=[first, blocked])
        startup = asyncio.create_task(registry.start())
        await blocked.entered.wait()

        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup

        assert first.closes == 1
        assert blocked.closes == 1

    asyncio.run(run_test())


def test_async_tool_registry_retries_failed_start_rollback():
    class FailingProvider(AsyncToolProvider):
        name = "failing"

        def __init__(self) -> None:
            self.closes = 0

        async def start(self):
            raise ValueError("startup failed")

        async def close(self):
            self.closes += 1
            if self.closes == 1:
                raise RuntimeError("cleanup failed")

    async def run_test() -> None:
        provider = FailingProvider()
        registry = ToolRegistry(async_providers=[provider])

        with pytest.raises(ValueError, match="startup failed") as raised:
            await registry.start()

        assert provider.closes == 1
        assert raised.value.__notes__ == [
            "failed to roll back 1 async tool provider(s)"
        ]
        with pytest.raises(RuntimeError, match="still require cleanup"):
            await registry.start()

        await registry.close()
        assert provider.closes == 2

    asyncio.run(run_test())


def test_async_tool_registry_bounds_each_close_and_continues_cleanup():
    class RecordingProvider(AsyncToolProvider):
        def __init__(self, name: str, *, block_close: bool = False) -> None:
            self.name = name
            self.block_close = block_close
            self.closes = 0

        async def start(self):
            return []

        async def close(self):
            self.closes += 1
            if self.block_close and self.closes == 1:
                await asyncio.Event().wait()

    async def run_test() -> None:
        quick = RecordingProvider("quick")
        blocked = RecordingProvider("blocked", block_close=True)
        registry = ToolRegistry(
            async_providers=[quick, blocked],
            provider_close_timeout_seconds=0.01,
        )
        await registry.start()

        with pytest.raises(RuntimeError, match="failed to close 1"):
            await registry.close()

        assert blocked.closes == 1
        assert quick.closes == 1
        with pytest.raises(RuntimeError, match="still require cleanup"):
            await registry.start()

        await registry.close()
        assert blocked.closes == 2
        assert quick.closes == 1

    asyncio.run(run_test())


def test_mcp_discovery_bounds_pages_bytes_and_total_time(tmp_path):
    class EndlessSession:
        def __init__(self) -> None:
            self.calls = 0

        async def list_tools(self, cursor):
            self.calls += 1
            return types.ListToolsResult(
                tools=[],
                nextCursor=f"cursor_{self.calls}",
            )

    class OversizedSession:
        def __init__(self) -> None:
            self.calls = 0

        async def list_tools(self, cursor):
            self.calls += 1
            return types.ListToolsResult(
                tools=[
                    types.Tool(
                        name=f"large_{self.calls}",
                        description="x" * 600,
                        inputSchema={"type": "object"},
                    )
                ],
                nextCursor="more" if self.calls == 1 else None,
            )

    class SlowSession:
        async def list_tools(self, cursor):
            await asyncio.sleep(60)

    class LargeCursorSession:
        async def list_tools(self, cursor):
            return types.ListToolsResult(tools=[], nextCursor="x" * 2048)

    async def run_test() -> None:
        paged = MCPStdioProvider(
            "paged",
            MCPServerSettings(command="unused", max_pages=2),
            workspace_cwd=tmp_path,
        )
        endless = EndlessSession()
        with pytest.raises(RuntimeError, match="max_pages=2"):
            await paged._list_tools(endless)
        assert endless.calls == 2

        bounded = MCPStdioProvider(
            "bounded",
            MCPServerSettings(command="unused", max_discovery_bytes=1024),
            workspace_cwd=tmp_path,
        )
        oversized = OversizedSession()
        with pytest.raises(RuntimeError, match="max_discovery_bytes=1024"):
            await bounded._list_tools(oversized)
        assert oversized.calls == 2

        with pytest.raises(RuntimeError, match="max_discovery_bytes=1024"):
            await bounded._list_tools(LargeCursorSession())

        timed = MCPStdioProvider(
            "timed",
            MCPServerSettings(command="unused").model_copy(
                update={"timeout_seconds": 0.01}
            ),
            workspace_cwd=tmp_path,
        )
        with pytest.raises(TimeoutError):
            await timed._list_tools(SlowSession())

    asyncio.run(run_test())


def test_mcp_provider_cancellation_stops_owner_task(tmp_path, monkeypatch):
    async def run_test() -> None:
        provider = MCPStdioProvider(
            "cancelled",
            MCPServerSettings(command="unused"),
            workspace_cwd=tmp_path,
        )
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def fake_owner(ready, stop_event):
            entered.set()
            try:
                await stop_event.wait()
            finally:
                cleaned.set()

        monkeypatch.setattr(provider, "_run_owner", fake_owner)
        startup = asyncio.create_task(provider.start())
        await entered.wait()
        startup.cancel()

        with pytest.raises(asyncio.CancelledError):
            await startup

        assert cleaned.is_set()
        assert provider._owner_task is None

    asyncio.run(run_test())


def test_mcp_stdio_provider_reports_startup_failure(tmp_path):
    async def run_test():
        provider = MCPStdioProvider(
            "missing",
            MCPServerSettings(command="codecraft-command-that-does-not-exist"),
            workspace_cwd=tmp_path,
        )

        with pytest.raises(MCPConnectionError) as raised:
            await provider.start()

        assert raised.value.code == "mcp_connection_failed"
        assert raised.value.metadata == {"mcp_server": "missing"}
        assert provider._owner_task is None

        with pytest.raises(MCPConnectionError):
            await provider.start()
        assert provider._owner_task is None

    asyncio.run(run_test())


def test_mcp_stdio_provider_can_close_from_a_different_task(tmp_path):
    async def run_test() -> None:
        fixture = Path(__file__).parent / "fixtures" / "mcp_test_server.py"
        provider = MCPStdioProvider(
            "cross_task",
            MCPServerSettings(command=sys.executable, args=[str(fixture)]),
            workspace_cwd=tmp_path,
        )

        tools = await asyncio.create_task(provider.start())
        assert {tool.name for tool in tools} == {
            "mcp__cross_task__add",
            "mcp__cross_task__echo",
        }
        await asyncio.create_task(provider.close())
        assert provider._owner_task is None

    asyncio.run(run_test())


def test_mcp_provider_close_can_be_bounded_and_retried(tmp_path):
    async def run_test() -> None:
        provider = MCPStdioProvider(
            "bounded_close",
            MCPServerSettings(command="unused"),
            workspace_cwd=tmp_path,
        )
        stop_event = asyncio.Event()
        release = asyncio.Event()

        async def fake_owner() -> None:
            await stop_event.wait()
            await release.wait()

        owner = asyncio.create_task(fake_owner())
        provider._owner_task = owner
        provider._stop_event = stop_event

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await provider.close()

        assert stop_event.is_set()
        assert not owner.done()
        assert provider._owner_task is owner

        release.set()
        await provider.close()
        assert owner.done()
        assert provider._owner_task is None

    asyncio.run(run_test())


def test_mcp_tool_timeout_marks_remote_outcome_unknown(tmp_path):
    class SlowSession:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(60)

    async def run_test() -> None:
        remote = types.Tool(
            name="mutate",
            inputSchema={"type": "object", "additionalProperties": False},
        )
        tool = MCPTool(
            session=SlowSession(),
            server_name="remote",
            remote_tool=remote,
            local_name="mcp__remote__mutate",
            effects={ToolEffect.EXTERNAL},
            requires_approval=True,
            timeout_seconds=0.01,
        )
        call = ToolCall(
            call_id="call_timeout",
            name=tool.name,
            arguments={},
        )

        result = await tool.arun(
            tool.args_schema.model_validate({}),
            ToolContext(context=_turn_context(tmp_path), call=call),
        )

        assert result.error == "mcp_tool_timeout"
        assert result.metadata["outcome_unknown"] is True
        assert result.metadata["retry_safe"] is False
        assert "may still have completed" in (result.suggestion or "")

    asyncio.run(run_test())


def test_mcp_stdio_provider_discovers_and_executes_through_tool_runner(tmp_path):
    async def run_test():
        fixture = Path(__file__).parent / "fixtures" / "mcp_test_server.py"
        settings = MCPServerSettings(
            command=sys.executable,
            args=[str(fixture)],
            tools={
                "add": MCPToolPolicySettings(
                    effects={"read_only"},
                    requires_approval=False,
                )
            },
        )
        provider = MCPStdioProvider("calculator", settings, workspace_cwd=tmp_path)
        registry = ToolRegistry(async_providers=[provider])
        await registry.start()
        try:
            add = registry.get("mcp__calculator__add")
            echo = registry.get("mcp__calculator__echo")
            assert add.effects == {ToolEffect.READ_ONLY}
            assert add.requires_approval is False
            assert add.annotations["readOnlyHint"] is True
            assert echo.effects == {ToolEffect.NETWORK, ToolEffect.EXTERNAL}
            assert echo.requires_approval is True

            runner = ToolRunner(
                registry,
                approval_manager=ApprovalManager(),
            )
            events = [
                event
                async for event in runner.run(
                    ToolCall(
                        call_id="call_add",
                        name="mcp__calculator__add",
                        arguments={"a": 2, "b": 3},
                    ),
                    _turn_context(tmp_path),
                )
            ]

            assert [event.type for event in events] == [
                RuntimeEventType.TOOL_CALL_STARTED,
                RuntimeEventType.TOOL_CALL_FINISHED,
            ]
            result = events[-1].payload["result"]
            assert result["success"] is True
            assert result["data"]["structured_content"] == {"sum": 5}
            assert result["metadata"]["mcp_server"] == "calculator"
            assert result["metadata"]["mcp_tool"] == "add"

            invalid = [
                event
                async for event in runner.run(
                    ToolCall(
                        call_id="call_invalid",
                        name="mcp__calculator__add",
                        arguments={"a": "two", "b": 3},
                    ),
                    _turn_context(tmp_path),
                )
            ][-1].payload["result"]
            assert invalid["success"] is False
            assert "validation" in invalid["content"].lower()
        finally:
            await registry.close()

    asyncio.run(run_test())
