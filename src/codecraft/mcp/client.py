from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import dataclass
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

from codecraft.core.async_utils import finish_task_before_cancelling
from codecraft.core.errors import CodecraftError
from codecraft.mcp.config import MCPServerSettings
from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.tool.base import BaseTool, ToolContext
from codecraft.tool.provider import AsyncToolProvider

_TOOL_NAME_CHARACTER = re.compile(r"[^A-Za-z0-9_-]")


class MCPConnectionError(CodecraftError):
    """MCP server 建连、发现或关闭生命周期失败。"""


@dataclass(frozen=True, slots=True)
class _StartedMCP:
    """Owner 初始化完成后一次性交付的 Session、server info 与适配工具。"""

    session: ClientSession
    server_info: dict[str, Any]
    tools: tuple[MCPTool, ...]


class MCPStdioProvider(AsyncToolProvider):
    """拥有一个宿主 stdio MCP 子进程/Session 并将远程工具适配进 Registry。

    Provider 负责生命周期、握手、发现预算和协议适配；它不为 server 建立 Docker
    或原生沙箱。ToolRegistry 发布后的审批/effect 检查发生在每次工具调用边界，
    不能约束一个已经运行的 server 进程，因此配置来源必须先经过用户信任校验。
    """

    def __init__(
        self,
        server_name: str,
        settings: MCPServerSettings,
        *,
        workspace_cwd: Path,
    ) -> None:
        """保存服务器配置和 workspace 基准，初始化尚未启动的 owner 状态。"""
        self.server_name = server_name
        self.name = f"mcp:{server_name}"
        self.settings = settings
        self.workspace_cwd = workspace_cwd
        self._owner_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._server_info: dict[str, Any] = {}

    async def start(self) -> tuple[BaseTool, ...]:
        """启动 owner task，等待初始化完成后才向 ToolRegistry 发布工具。

        Returns:
            名称安全、schema 已验证且附有本地治理 effects 的 MCPTool 元组。

        Raises:
            RuntimeError: Provider 已启动。
            MCPConnectionError: 进程、握手、发现或失败清理异常。

        Cancellation:
            ready 等待被 shield；调用方取消时显式取消 owner 并完成退出栈清理，
            再传播 CancelledError，不留下 stdio 子进程。
        """
        if self._owner_task is not None:
            raise RuntimeError(f"MCP server is already started: {self.server_name}")

        ready: asyncio.Future[_StartedMCP] = asyncio.get_running_loop().create_future()
        stop_event = asyncio.Event()
        owner = asyncio.create_task(
            self._run_owner(ready, stop_event),
            name=f"codecraft-{self.name}",
        )
        self._owner_task = owner
        self._stop_event = stop_event
        try:
            started = await asyncio.shield(ready)
        except BaseException as exc:
            cleanup_error = await self._cleanup_failed_start(
                owner,
                ready,
                stop_event,
                exc,
            )
            if not isinstance(exc, Exception):
                raise
            detail = f"{type(exc).__name__}: {exc}"
            if cleanup_error is not None:
                detail += f"; cleanup {type(cleanup_error).__name__}: {cleanup_error}"
            raise MCPConnectionError(
                f"Could not start MCP server '{self.server_name}'.",
                code="mcp_connection_failed",
                suggestion=detail,
                metadata={"mcp_server": self.server_name},
            ) from exc

        self._session = started.session
        self._server_info = started.server_info
        return started.tools

    async def _cleanup_failed_start(
        self,
        owner: asyncio.Task[None],
        ready: asyncio.Future[_StartedMCP],
        stop_event: asyncio.Event,
        original: BaseException,
    ) -> Exception | None:
        """根据原始取消/异常停止 owner，消费 Future 异常并清空生命周期状态。"""
        if isinstance(original, asyncio.CancelledError):
            owner.cancel()
        else:
            stop_event.set()
        cleanup_error: Exception | None = None
        try:
            try:
                await finish_task_before_cancelling(owner)
            except asyncio.CancelledError:
                if not isinstance(original, asyncio.CancelledError):
                    raise
                if ready.done() and not ready.cancelled():
                    ready.exception()
            except Exception as exc:
                if exc is not original:
                    cleanup_error = exc
        finally:
            self._clear_owner()
        return cleanup_error

    async def close(self) -> None:
        """通知 owner 退出并等待其在原任务内关闭 Session/stdio AsyncExitStack。

        外层取消不会取消被 shield 的 owner；若 owner 自身意外取消，则翻译成
        mcp_close_failed。正常、失败和可判定异常都会清理本地引用。
        """
        owner = self._owner_task
        stop_event = self._stop_event
        if owner is None or stop_event is None:
            return
        stop_event.set()
        try:
            await asyncio.shield(owner)
        except asyncio.CancelledError:
            if owner.done():
                self._clear_owner()
                raise MCPConnectionError(
                    f"MCP server '{self.server_name}' stopped unexpectedly.",
                    code="mcp_close_failed",
                    metadata={"mcp_server": self.server_name},
                )
            raise
        except Exception as exc:
            self._clear_owner()
            raise MCPConnectionError(
                f"Could not close MCP server '{self.server_name}'.",
                code="mcp_close_failed",
                suggestion=f"{type(exc).__name__}: {exc}",
                metadata={"mcp_server": self.server_name},
            ) from exc
        self._clear_owner()

    async def _run_owner(
        self,
        ready: asyncio.Future[_StartedMCP],
        stop_event: asyncio.Event,
    ) -> None:
        """在单一 task 内进入、持有并退出 stdio 与 ClientSession 上下文。

        初始化/分页发现受统一 timeout 保护；ready 交付后 owner 等 stop_event，
        finally 再限时 aclose。保持 enter/exit task affinity 避免 anyio cancel
        scope 跨任务退出错误。
        """
        stack = AsyncExitStack()
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
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
                initialized = await session.initialize()
                remote_tools = await self._list_tools(session)
                tools = self._adapt_tools(session, remote_tools)
                server_info = initialized.serverInfo.model_dump(mode="json")
                for tool in tools:
                    tool.server_info = server_info
            ready.set_result(
                _StartedMCP(
                    session=session,
                    server_info=server_info,
                    tools=tools,
                )
            )
            await stop_event.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise
        finally:
            async with asyncio.timeout(self.settings.timeout_seconds):
                await stack.aclose()

    def _clear_owner(self) -> None:
        """清除 owner、stop event、Session 和 server metadata 引用。"""
        self._owner_task = None
        self._stop_event = None
        self._session = None
        self._server_info = {}

    async def _list_tools(self, session: ClientSession) -> tuple[types.Tool, ...]:
        """有界遍历远程工具分页，限制页数、工具数、总 JSON bytes 与 cursor。

        重复 cursor 会拒绝，防止恶意/错误服务器无限循环；每页在追加前检查
        max_tools，工具 schema 和 cursor 都计入 discovery byte budget。
        """
        tools: list[types.Tool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        discovery_bytes = 0
        async with asyncio.timeout(self.settings.timeout_seconds):
            for _ in range(self.settings.max_pages):
                page = await session.list_tools(cursor)
                if len(page.tools) > self.settings.max_tools - len(tools):
                    raise RuntimeError(
                        f"MCP server {self.server_name} exceeds max_tools="
                        f"{self.settings.max_tools}"
                    )
                discovery_bytes += sum(_tool_size(tool) for tool in page.tools)
                if discovery_bytes > self.settings.max_discovery_bytes:
                    raise RuntimeError(
                        f"MCP server {self.server_name} exceeds "
                        f"max_discovery_bytes={self.settings.max_discovery_bytes}"
                    )
                tools.extend(page.tools)
                cursor = page.nextCursor
                if cursor is None:
                    return tuple(tools)
                discovery_bytes += len(cursor.encode("utf-8"))
                if discovery_bytes > self.settings.max_discovery_bytes:
                    raise RuntimeError(
                        f"MCP server {self.server_name} exceeds "
                        f"max_discovery_bytes={self.settings.max_discovery_bytes}"
                    )
                if cursor in seen_cursors:
                    raise RuntimeError(
                        f"MCP server {self.server_name} repeated pagination cursor"
                    )
                seen_cursors.add(cursor)
            raise RuntimeError(
                f"MCP server {self.server_name} exceeds max_pages="
                f"{self.settings.max_pages}"
            )

    def _adapt_tools(
        self,
        session: ClientSession,
        remote_tools: tuple[types.Tool, ...],
    ) -> tuple[MCPTool, ...]:
        """映射远程名称/schema/annotations，并应用本地而非远端声明的治理策略。

        清洗/截断后的本地名在同一服务器内冲突会拒绝整个启动，避免模型调用
        一个名字却无法确定实际远程目标。
        """
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
        """从 MCP SDK 默认环境加显式 allowlist 构造子进程环境。"""
        environment = get_default_environment()
        for name in self.settings.env_allowlist:
            value = os.environ.get(name)
            if value is not None:
                environment[name] = value
        return environment

    def _cwd(self) -> Path:
        """相对 cwd 基于 workspace 解析，并要求启动时已存在且是目录。"""
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
    """把一个 MCP Tool 的 JSON Schema、调用与多模态结果适配为 BaseTool。"""

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
        """绑定共享 Session、远程身份、本地策略、schema 和 timeout。"""
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
        """限时调用远程工具并归一化文本、资源、多媒体和 structured content。

        超时标记 outcome_unknown/retry_safe=False，因为远程副作用可能已完成；
        MCP isError 决定 ToolResult success，而非是否存在文本。structuredContent
        在没有文本 block 时作为稳定排序 JSON 提供给模型。
        """
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
                suggestion=(
                    "The remote operation may still have completed; inspect state before "
                    "retrying."
                ),
                metadata={
                    **self._metadata(),
                    "outcome_unknown": True,
                    "retry_safe": False,
                },
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
        """返回服务器/远程工具/annotations/握手信息组成的审计身份。"""
        return {
            "mcp_server": self.server_name,
            "mcp_tool": self.remote_name,
            "mcp_annotations": self.annotations,
            "mcp_server_info": self.server_info,
        }


def mcp_tool_name(server_name: str, remote_name: str) -> str:
    """生成不超过 64 字符的稳定本地 Tool 名，长名称附远程名摘要。

    Example:
        >>> mcp_tool_name("repo", "search/files")
        'mcp__repo__search_files'
    """
    sanitized = _TOOL_NAME_CHARACTER.sub("_", remote_name).strip("_") or "tool"
    candidate = f"mcp__{server_name}__{sanitized}"
    if len(candidate) <= 64:
        return candidate
    digest = sha256(remote_name.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55]}_{digest}"


def _tool_size(tool: types.Tool) -> int:
    """按紧凑 Unicode JSON 的 UTF-8 bytes 计算远程工具发现成本。"""
    serialized = json.dumps(
        tool.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(serialized.encode("utf-8"))


def mcp_args_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """从远程 Draft 2020-12 JSON Schema 创建动态 Pydantic 参数模型。

    Schema 自身先经 meta-schema 校验；实例校验保留 MCP 原 schema 语义，
    Pydantic 允许 extra 以免自行施加远程未声明的限制；model_json_schema
    返回深拷贝原 schema，使暴露给模型的协议不被 Pydantic 重写。
    """
    input_schema = deepcopy(schema)
    Draft202012Validator.check_schema(input_schema)
    validator = Draft202012Validator(input_schema)

    class MCPArguments(BaseModel):
        """闭包绑定远程 JSON Schema 的动态工具参数模型。"""

        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def validate_mcp_schema(cls, value: Any) -> Any:
            """运行 JSON Schema 校验并用首个路径明确的错误拒绝参数。"""
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
            """忽略 Pydantic 生成选项并返回远程 inputSchema 深拷贝。"""
            return deepcopy(input_schema)

    MCPArguments.__name__ = f"{name}_arguments"
    return MCPArguments


def _format_mcp_content(
    content: list[types.ContentBlock],
) -> tuple[str, list[dict[str, Any]]]:
    """把 MCP content blocks 转为模型文本和不含二进制正文的结构摘要。

    文本与 text resource 保留正文；resource link 变成 URI 提示；blob/image/
    audio 只记录 MIME 与解码后 bytes，避免 base64 大对象直接进入模型文本。
    """
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
    """严格解码 base64 后返回字节数，无效数据安全返回 0。"""
    try:
        return len(base64.b64decode(value, validate=True))
    except ValueError:
        return 0
