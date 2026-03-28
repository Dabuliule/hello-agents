"""异步工具执行器。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

from core.exceptions import ToolException
from .registry import ToolRegistry


@dataclass(frozen=True)
class ToolCall:
    """单次工具调用描述。"""

    name: str
    params: dict[str, Any] | None = None
    key: str | None = None


class AsyncToolExecutor:
    """基于 asyncio 的工具执行器，支持并发调用。"""

    def __init__(self, registry: ToolRegistry, max_concurrency: int = 5):
        if max_concurrency < 1:
            raise ValueError("max_concurrency 必须大于 0")
        self.registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def execute(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """异步执行单个工具。"""
        return await self._run_in_thread(name, params)

    async def execute_many(self, calls: Iterable[ToolCall]) -> list[Any]:
        """并发执行多个工具，保持输入顺序返回结果。"""
        tasks = [asyncio.create_task(self._run_in_thread(call.name, call.params)) for call in calls]
        return await asyncio.gather(*tasks)

    async def execute_map(self, calls: Iterable[ToolCall]) -> dict[str, Any]:
        """并发执行多个工具，按 key 返回结果。"""
        tasks: list[tuple[str, asyncio.Task[Any]]] = []
        for index, call in enumerate(calls):
            key = call.key or f"call_{index}"
            tasks.append((key, asyncio.create_task(self._run_in_thread(call.name, call.params))))

        results: dict[str, Any] = {}
        for key, task in tasks:
            results[key] = await task
        return results

    async def _run_in_thread(self, name: str, params: dict[str, Any] | None) -> Any:
        """在线程池中执行同步工具调用。"""
        async with self._semaphore:
            try:
                return await asyncio.to_thread(self.registry.execute, name, params)
            except ToolException:
                raise
            except Exception as exc:
                raise ToolException(f"工具 {name} 执行失败: {exc}") from exc
