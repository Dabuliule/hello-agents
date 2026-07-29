from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.turn_context import TurnContext
from codecraft.mcp.client import MCPStdioProvider
from codecraft.mcp.config import MCPServerSettings, MCPToolPolicySettings
from codecraft.mcp.server import create_repository_mcp_server
from codecraft.sandbox import SandboxMode
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.tool import ToolCall, ToolEffect
from codecraft.tool import ToolRegistry
from codecraft.tool.runner import ToolRunner


def _turn_context(tmp_path) -> TurnContext:
    return TurnContext(
        session_id="ses_mcp_server",
        turn_id="turn_mcp_server",
        cwd=tmp_path,
        model="none",
        model_provider="test",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.READ_ONLY,
        network_access=False,
        available_tools=[],
        max_tool_calls=1,
        max_tool_output_chars=80_000,
        created_at=datetime.now(UTC),
    )


def test_repository_mcp_server_exposes_read_only_tool_and_resources(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Keep changes focused.\n", encoding="utf-8")
    server = create_repository_mcp_server(
        tmp_path,
        codecraft_home=tmp_path / ".codecraft-home",
    )

    async def inspect_server():
        tools = await server.list_tools()
        resources = await server.list_resources()
        metadata = list(await server.read_resource("codecraft://workspace/metadata"))
        instructions = list(
            await server.read_resource("codecraft://workspace/instructions")
        )
        return tools, resources, metadata, instructions

    tools, resources, metadata, instructions = asyncio.run(inspect_server())

    assert [tool.name for tool in tools] == ["search_repository"]
    assert tools[0].annotations.readOnlyHint is True
    assert tools[0].annotations.openWorldHint is False
    assert {str(resource.uri) for resource in resources} == {
        "codecraft://workspace/instructions",
        "codecraft://workspace/metadata",
    }
    metadata_value = json.loads(metadata[0].content)
    assert metadata_value["workspace"] == str(tmp_path)
    assert metadata_value["index_available"] is False
    assert metadata_value["retrievers"] == ["scan", "lexical", "symbol"]
    assert "Keep changes focused." in instructions[0].content


def test_codecraft_mcp_client_and_server_interoperate_over_stdio(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text(
        "SPECIAL_CONTEXT_TOKEN = 'ready'\n",
        encoding="utf-8",
    )

    async def run_test():
        settings = MCPServerSettings(
            command=sys.executable,
            args=[
                "-m",
                "codecraft.cli.app",
                "mcp-server",
                "--workspace",
                str(tmp_path),
                "--codecraft-home",
                str(tmp_path / ".codecraft-home"),
            ],
            tools={
                "search_repository": MCPToolPolicySettings(
                    effects={"read_only"},
                    requires_approval=False,
                )
            },
        )
        provider = MCPStdioProvider("codecraft", settings, workspace_cwd=tmp_path)
        registry = ToolRegistry(async_providers=[provider])
        await registry.start()
        try:
            tool = registry.get("mcp__codecraft__search_repository")
            assert tool.effects == {ToolEffect.READ_ONLY}
            assert tool.annotations["readOnlyHint"] is True

            events = [
                event
                async for event in ToolRunner(
                    registry,
                    approval_manager=ApprovalManager(),
                ).run(
                    ToolCall(
                        call_id="call_search",
                        name="mcp__codecraft__search_repository",
                        arguments={
                            "query": "SPECIAL_CONTEXT_TOKEN",
                            "path": "src",
                            "strategy": "scan",
                            "max_results": 5,
                        },
                    ),
                    _turn_context(tmp_path),
                )
            ]
            escaped_events = [
                event
                async for event in ToolRunner(
                    registry,
                    approval_manager=ApprovalManager(),
                ).run(
                    ToolCall(
                        call_id="call_escape",
                        name="mcp__codecraft__search_repository",
                        arguments={"query": "secret", "path": ".."},
                    ),
                    _turn_context(tmp_path),
                )
            ]
        finally:
            await registry.close()
        return events, escaped_events

    events, escaped_events = asyncio.run(run_test())

    assert [event.type for event in events] == [
        RuntimeEventType.TOOL_CALL_STARTED,
        RuntimeEventType.TOOL_CALL_FINISHED,
    ]
    result = events[-1].payload["result"]
    assert result["success"] is True
    structured = result["data"]["structured_content"]
    assert structured["match_count"] == 1
    assert structured["retriever"] == "scan"
    assert structured["matches"][0]["path"] == "src/app.py"
    assert "SPECIAL_CONTEXT_TOKEN" in structured["matches"][0]["snippet"]
    escaped = escaped_events[-1].payload["result"]
    assert escaped["success"] is False
    assert escaped["error"] == "mcp_tool_error"
    assert "outside workspace" in escaped["content"]
