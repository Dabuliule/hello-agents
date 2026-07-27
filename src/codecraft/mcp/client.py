from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from datetime import timedelta
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode

from codecraft.core.errors import CodecraftError
from codecraft.mcp.config import MCPServerSettings
from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.tool.base import BaseTool, ToolContext
from codecraft.tool.provider import AsyncToolProvider

_TOOL_NAME_CHARACTER = re.compile(r"[^A-Za-z0-9_-]")


class MCPConnectionError(CodecraftError):
    pass


class MCPStdioProvider(AsyncToolProvider):
    def __init__(
        self,
        server_name: str,
        settings: MCPServerSettings,
        *,
        workspace_cwd: Path,
    ) -> None:
        self.server_name = server_name
        self.name = f"mcp:{server_name}"
        self.settings = settings
        self.workspace_cwd = workspace_cwd
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._server_info: dict[str, Any] = {}

    async def start(self) -> tuple[BaseTool, ...]:
        if self._session is not None:
            raise RuntimeError(f"MCP server is already started: {self.server_name}")

        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=self.settings.command,
                        args=self.settings.args,
                        env=self._environment(),
                        cwd=self._cwd(),
                    ),
                    errlog=sys.stderr,
                )
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(
                        seconds=self.settings.timeout_seconds
                    ),
                    client_info=types.Implementation(
                        name="codecraft",
                        version=version("codecraft"),
                    ),
                )
            )
            initialized = await asyncio.wait_for(
                session.initialize(), timeout=self.settings.timeout_seconds
            )
            remote_tools = await self._list_tools(session)
            tools = self._adapt_tools(session, remote_tools)
        except Exception as exc:
            await stack.aclose()
            raise MCPConnectionError(
                f"Could not start MCP server '{self.server_name}'.",
                code="mcp_connection_failed",
                suggestion=f"{type(exc).__name__}: {exc}",
                metadata={"mcp_server": self.server_name},
            ) from exc

        self._stack = stack
        self._session = session
        self._server_info = initialized.serverInfo.model_dump(mode="json")
        for tool in tools:
            tool.server_info = self._server_info
        return tools

    async def close(self) -> None:
        stack = self._stack
        self._stack = None
        self._session = None
        self._server_info = {}
        if stack is not None:
            await stack.aclose()

    async def _list_tools(self, session: ClientSession) -> tuple[types.Tool, ...]:
        tools: list[types.Tool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            page = await asyncio.wait_for(
                session.list_tools(cursor), timeout=self.settings.timeout_seconds
            )
            tools.extend(page.tools)
            if len(tools) > self.settings.max_tools:
                raise RuntimeError(
                    f"MCP server {self.server_name} exceeds max_tools="
                    f"{self.settings.max_tools}"
                )
            cursor = page.nextCursor
            if cursor is None:
                return tuple(tools)
            if cursor in seen_cursors:
                raise RuntimeError(
                    f"MCP server {self.server_name} repeated pagination cursor"
                )
            seen_cursors.add(cursor)

    def _adapt_tools(
        self,
        session: ClientSession,
        remote_tools: tuple[types.Tool, ...],
    ) -> tuple[MCPTool, ...]:
        tools: list[MCPTool] = []
        local_names: set[str] = set()
        for remote in remote_tools:
            local_name = mcp_tool_name(self.server_name, remote.name)
            if local_name in local_names:
                raise RuntimeError(
                    f"MCP tools map to duplicate local name: {local_name}"
                )
            local_names.add(local_name)
            policy = self.settings.policy_for(remote.name)
            tools.append(
                MCPTool(
                    session=session,
                    server_name=self.server_name,
                    remote_tool=remote,
                    local_name=local_name,
                    effects={ToolEffect(effect) for effect in policy.effects},
                    requires_approval=policy.requires_approval,
                    timeout_seconds=self.settings.timeout_seconds,
                )
            )
        return tuple(tools)

    def _environment(self) -> dict[str, str]:
        environment = get_default_environment()
        for name in self.settings.env_allowlist:
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value
        return environment

    def _cwd(self) -> Path:
        configured = self.settings.cwd
        cwd = (
            self.workspace_cwd
            if configured is None
            else configured
            if configured.is_absolute()
            else self.workspace_cwd / configured
        ).resolve()
        if not cwd.exists() or not cwd.is_dir():
            raise ValueError(f"MCP server cwd must be a directory: {cwd}")
        return cwd


class MCPTool(BaseTool):
    def __init__(
        self,
        *,
        session: ClientSession,
        server_name: str,
        remote_tool: types.Tool,
        local_name: str,
        effects: set[ToolEffect],
        requires_approval: bool,
        timeout_seconds: int,
    ) -> None:
        self.session = session
        self.server_name = server_name
        self.remote_name = remote_tool.name
        self.name = local_name
        self.description = remote_tool.description or f"MCP tool {remote_tool.name}."
        self.args_schema = mcp_args_model(local_name, remote_tool.inputSchema)
        self.effects = effects
        self.requires_approval = requires_approval
        self.timeout_seconds = timeout_seconds
        self.annotations = (
            remote_tool.annotations.model_dump(mode="json")
            if remote_tool.annotations is not None
            else None
        )
        self.server_info: dict[str, Any] = {}

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(
                    self.remote_name,
                    arguments=args.model_dump(mode="json"),
                ),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                content="MCP tool call timed out.",
                error="mcp_tool_timeout",
                metadata=self._metadata(),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                content="MCP tool call failed.",
                error="mcp_tool_error",
                suggestion=f"{type(exc).__name__}: {exc}",
                metadata=self._metadata(),
            )

        content, blocks = _format_mcp_content(result.content)
        structured = result.structuredContent
        if not content and structured is not None:
            content = json.dumps(structured, ensure_ascii=False, sort_keys=True)
        success = not result.isError
        return ToolResult(
            success=success,
            content=content
            or ("MCP tool completed." if success else "MCP tool failed."),
            data={
                "structured_content": structured,
                "content_blocks": blocks,
            },
            error=None if success else "mcp_tool_error",
            metadata=self._metadata(),
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "mcp_server": self.server_name,
            "mcp_tool": self.remote_name,
            "mcp_annotations": self.annotations,
            "mcp_server_info": self.server_info,
        }


def mcp_tool_name(server_name: str, remote_name: str) -> str:
    sanitized = _TOOL_NAME_CHARACTER.sub("_", remote_name).strip("_") or "tool"
    candidate = f"mcp__{server_name}__{sanitized}"
    if len(candidate) <= 64:
        return candidate
    digest = sha256(remote_name.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55]}_{digest}"


def mcp_args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    input_schema = deepcopy(schema)
    Draft202012Validator.check_schema(input_schema)
    validator = Draft202012Validator(input_schema)

    class MCPArguments(BaseModel):
        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def validate_mcp_schema(cls, value: Any) -> Any:
            errors = sorted(
                validator.iter_errors(value), key=lambda error: list(error.path)
            )
            if errors:
                error = errors[0]
                location = ".".join(str(part) for part in error.path) or "arguments"
                raise ValueError(f"{location}: {error.message}")
            return value

        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = "#/$defs/{model}",
            schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
            mode: JsonSchemaMode = "validation",
            *,
            union_format: Literal["any_of", "primitive_type_array"] = "any_of",
        ) -> dict[str, Any]:
            return deepcopy(input_schema)

    MCPArguments.__name__ = f"{name}_arguments"
    return MCPArguments


def _format_mcp_content(
    content: list[types.ContentBlock],
) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, types.TextContent):
            text_parts.append(block.text)
            blocks.append({"type": "text", "characters": len(block.text)})
        elif isinstance(block, types.ResourceLink):
            text_parts.append(f"[resource {block.name}: {block.uri}]")
            blocks.append(
                {"type": "resource_link", "name": block.name, "uri": str(block.uri)}
            )
        elif isinstance(block, types.EmbeddedResource):
            resource = block.resource
            if isinstance(resource, types.TextResourceContents):
                text_parts.append(f"[{resource.uri}]\n{resource.text}")
                blocks.append(
                    {
                        "type": "embedded_text_resource",
                        "uri": str(resource.uri),
                        "characters": len(resource.text),
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "embedded_blob_resource",
                        "uri": str(resource.uri),
                        "bytes": _base64_size(resource.blob),
                        "mime_type": resource.mimeType,
                    }
                )
                text_parts.append(f"[binary resource: {resource.uri}]")
        elif isinstance(block, (types.ImageContent, types.AudioContent)):
            kind = "image" if isinstance(block, types.ImageContent) else "audio"
            blocks.append(
                {
                    "type": kind,
                    "bytes": _base64_size(block.data),
                    "mime_type": block.mimeType,
                }
            )
            text_parts.append(f"[{kind}: {block.mimeType}]")
    return "\n".join(text_parts), blocks


def _base64_size(value: str) -> int:
    try:
        return len(base64.b64decode(value, validate=True))
    except ValueError:
        return 0
