from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from codecraft.tool.base import BaseTool


class ToolProvider(ABC):
    """无需异步生命周期即可批量提供 Tool 的扩展接口。"""

    name: str

    @abstractmethod
    def tools(self) -> Iterable[BaseTool]:
        """返回待注册的工具集合。"""
        ...


class AsyncToolProvider(ABC):
    """启动时动态发现 Tool、关闭时释放外部资源的 Provider 接口。"""

    name: str

    @abstractmethod
    async def start(self) -> Iterable[BaseTool]:
        """建立连接/发现能力，并返回本生命周期拥有的工具。"""
        ...

    @abstractmethod
    async def close(self) -> None:
        """释放 start 获取的资源；应支持失败后再次调用重试。"""
        ...
