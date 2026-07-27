from __future__ import annotations

import asyncio
import builtins
from collections.abc import Iterable

from codecraft.core.errors import ToolNotFoundError
from codecraft.schema.tool import ToolSpec
from codecraft.tool.base import BaseTool
from codecraft.tool.provider import AsyncToolProvider, ToolProvider


class ToolRegistry:
    """按 tool name 管理所有可调用 tool。"""

    def __init__(
        self,
        tools: Iterable[BaseTool] | None = None,
        async_providers: Iterable[AsyncToolProvider] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._async_providers: dict[str, AsyncToolProvider] = {}
        self._provider_tool_names: set[str] = set()
        self._started = False
        self._lifecycle_lock = asyncio.Lock()
        for tool in tools or ():
            self.register(tool)
        for provider in async_providers or ():
            self.register_async_provider(provider)

    def register(self, tool: BaseTool) -> None:
        """注册单个 tool，并拒绝空名称或重复名称。"""
        name = tool.name.strip()
        if not name:
            raise ValueError("tool name must not be empty")

        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")

        self._tools[name] = tool

    def register_provider(self, provider: ToolProvider) -> None:
        """注册一个 provider 暴露出的全部 tool。"""
        for tool in provider.tools():
            self.register(tool)

    def register_async_provider(self, provider: AsyncToolProvider) -> None:
        if self._started:
            raise RuntimeError("cannot add an async tool provider after registry start")
        name = provider.name.strip()
        if not name:
            raise ValueError("async tool provider name must not be empty")
        if name in self._async_providers:
            raise ValueError(f"async tool provider already registered: {name}")
        self._async_providers[name] = provider

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return

            started: list[AsyncToolProvider] = []
            discovered: list[BaseTool] = []
            names = set(self._tools)
            try:
                for provider in self._async_providers.values():
                    provider_tools = tuple(await provider.start())
                    started.append(provider)
                    for tool in provider_tools:
                        name = tool.name.strip()
                        if not name:
                            raise ValueError("tool name must not be empty")
                        if name in names:
                            raise ValueError(f"tool already registered: {name}")
                        names.add(name)
                        discovered.append(tool)
            except Exception:
                for provider in reversed(started):
                    await provider.close()
                raise

            for tool in discovered:
                self.register(tool)
                self._provider_tool_names.add(tool.name)
            self._started = True

    async def close(self) -> None:
        async with self._lifecycle_lock:
            if not self._started:
                return
            for name in self._provider_tool_names:
                self._tools.pop(name, None)
            self._provider_tool_names.clear()

            errors: list[Exception] = []
            for provider in reversed(tuple(self._async_providers.values())):
                try:
                    await provider.close()
                except Exception as exc:
                    errors.append(exc)
            self._started = False
            if errors:
                raise RuntimeError(
                    f"failed to close {len(errors)} async tool provider(s)"
                ) from errors[0]

    def get(self, name: str) -> BaseTool:
        """按名称取 tool，不存在时抛出带 code 的业务异常。"""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(
                f"tool not found: {name}",
                code="tool_not_found",
                metadata={"tool": name},
            ) from exc

    def list(self) -> builtins.list[BaseTool]:
        return builtins.list(self._tools.values())

    def specs(self) -> builtins.list[ToolSpec]:
        """返回所有 tool 的模型可见描述。"""
        return [tool.spec() for tool in self.list()]

    def async_provider_names(self) -> builtins.list[str]:
        return builtins.list(self._async_providers)
