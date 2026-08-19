"""统一管理静态工具与需要异步发现的动态工具 Provider。"""

from __future__ import annotations

import asyncio
import builtins
from collections.abc import Iterable

from codecraft.core.errors import ToolNotFoundError
from codecraft.schema.tool import ToolSpec
from codecraft.tool.base import BaseTool
from codecraft.tool.provider import AsyncToolProvider, ToolProvider


class ToolRegistry:
    """按唯一名称管理当前 Runtime 可以暴露给模型的全部 Tool。

    构造时传入的内置 Tool 会立即进入 ``_tools``，不需要单独启动。MCP 等
    ``AsyncToolProvider`` 构造时只登记 Provider 本身；它们必须等 ``start()``
    建立连接、发现远端工具并完成全局重名校验后，工具才会原子加入 Registry。
    """

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
        """在生命周期开始前登记 Provider，但暂不连接或发布其动态工具。

        Args:
            provider: 具有稳定名称、可异步启动和关闭的工具来源。

        Raises:
            RuntimeError: Registry 已经开始启动或仍有失败 Provider 等待清理。
            ValueError: Provider 名称为空或与已登记 Provider 重复。
        """
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
        """完成动态 Provider 初始化，全部成功后才原子发布其工具。

        中途失败会按已启动逆序 close；未清理成功的 Provider 留在 pending，
        禁止再次 start，直到 close 重试完成，避免资源泄漏和重复工具曝光。

        对只有内置工具的 Registry，本方法除了把生命周期标记为 started 外几乎
        是 no-op。对 MCP Registry，它会依次启动 server、完成握手与工具发现，
        先在临时集合校验所有名称，最后统一写入 ``_tools``。这样模型不会看到
        “一部分 MCP 已加载、另一部分启动失败”的半成品工具目录。

        本方法幂等；已经成功启动后再次调用会直接返回。
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
