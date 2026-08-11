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
        provider_close_timeout_seconds: float = 10.0,
    ) -> None:
        """注册静态工具/异步 Provider，并初始化串行生命周期状态。

        Raises:
            ValueError: close timeout 非正或初始工具/Provider 重名。
        """
        if provider_close_timeout_seconds <= 0:
            raise ValueError("provider close timeout must be positive")
        self._tools: dict[str, BaseTool] = {}
        self._async_providers: dict[str, AsyncToolProvider] = {}
        self._provider_tool_names: set[str] = set()
        self._pending_provider_closes: list[AsyncToolProvider] = []
        self._started = False
        self._provider_close_timeout_seconds = provider_close_timeout_seconds
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
        """在生命周期开始前登记名称唯一的异步 Provider。"""
        if self._started or self._pending_provider_closes:
            raise RuntimeError(
                "cannot add an async tool provider after registry lifecycle begins"
            )
        name = provider.name.strip()
        if not name:
            raise ValueError("async tool provider name must not be empty")
        if name in self._async_providers:
            raise ValueError(f"async tool provider already registered: {name}")
        self._async_providers[name] = provider

    async def start(self) -> None:
        """串行启动 Provider，全部成功后才原子发布其工具。

        中途失败会按已启动逆序 close；未清理成功的 Provider 留在 pending，
        禁止再次 start，直到 close 重试完成，避免资源泄漏和重复工具曝光。
        """
        async with self._lifecycle_lock:
            if self._started:
                return
            if self._pending_provider_closes:
                raise RuntimeError(
                    "cannot start while async providers still require cleanup"
                )

            started: list[AsyncToolProvider] = []
            discovered: list[BaseTool] = []
            names = set(self._tools)
            try:
                for provider in self._async_providers.values():
                    started.append(provider)
                    provider_tools = tuple(await provider.start())
                    for tool in provider_tools:
                        name = tool.name.strip()
                        if not name:
                            raise ValueError("tool name must not be empty")
                        if name in names:
                            raise ValueError(f"tool already registered: {name}")
                        names.add(name)
                        discovered.append(tool)
            except BaseException as exc:
                self._pending_provider_closes = list(reversed(started))
                errors = await self._close_pending_providers()
                if errors:
                    exc.add_note(
                        f"failed to roll back {len(errors)} async tool provider(s)"
                    )
                raise

            for tool in discovered:
                self.register(tool)
                self._provider_tool_names.add(tool.name)
            self._started = True

    async def close(self) -> None:
        """撤下动态工具并逆序关闭 Provider；失败项保留以供下次 close 重试。"""
        async with self._lifecycle_lock:
            if not self._started and not self._pending_provider_closes:
                return
            if self._started:
                for name in self._provider_tool_names:
                    self._tools.pop(name, None)
                self._provider_tool_names.clear()
                self._pending_provider_closes = list(
                    reversed(tuple(self._async_providers.values()))
                )
                self._started = False

            errors = await self._close_pending_providers()
            if errors:
                raise RuntimeError(
                    f"failed to close {len(errors)} async tool provider(s)"
                ) from errors[0]

    async def _close_pending_providers(self) -> list[BaseException]:
        """逐个限时关闭 pending Provider，保存失败项并正确传播取消。"""
        errors: list[BaseException] = []
        cancellation: asyncio.CancelledError | None = None
        providers = self._pending_provider_closes
        self._pending_provider_closes = []
        for provider in providers:
            try:
                async with asyncio.timeout(self._provider_close_timeout_seconds):
                    await provider.close()
            except asyncio.CancelledError as exc:
                cancellation = exc
                self._pending_provider_closes.append(provider)
            except Exception as exc:
                errors.append(exc)
                self._pending_provider_closes.append(provider)
        if cancellation is not None:
            if errors:
                cancellation.add_note(
                    f"failed to close {len(errors)} additional async tool provider(s)"
                )
            raise cancellation
        return errors

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
        """按注册顺序返回当前可调用工具快照。"""
        return builtins.list(self._tools.values())

    def specs(self) -> builtins.list[ToolSpec]:
        """返回所有 tool 的模型可见描述。"""
        return [tool.spec() for tool in self.list()]

    def async_provider_names(self) -> builtins.list[str]:
        """按登记顺序返回异步 Provider 名称。"""
        return builtins.list(self._async_providers)
