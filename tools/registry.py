"""Tool 注册中心。"""

from __future__ import annotations

from typing import Any, Iterable, Literal

from core.exceptions import ToolException
from .base import Tool

ConflictPolicy = Literal["error", "overwrite", "ignore"]


class ToolRegistry:
    """管理 Tool 的注册、查找和执行。"""

    def __init__(self, tools: Iterable[Tool] | None = None, conflict: ConflictPolicy = "error"):
        self._tools: dict[str, Tool] = {}
        if tools:
            self.register_many(tools, conflict=conflict)

    def register(self, tool: Tool, conflict: ConflictPolicy = "error") -> None:
        """注册单个工具。"""
        if not isinstance(tool, Tool):
            raise ToolException("只能注册 Tool 实例")

        name = tool.name
        exists = name in self._tools
        if exists and conflict == "error":
            raise ToolException(f"工具已存在: {name}")
        if exists and conflict == "ignore":
            return
        self._tools[name] = tool

    def register_many(self, tools: Iterable[Tool], conflict: ConflictPolicy = "error") -> None:
        """批量注册工具。"""
        for tool in tools:
            self.register(tool, conflict=conflict)

    def unregister(self, name: str) -> None:
        """移除工具。"""
        if name not in self._tools:
            raise ToolException(f"工具不存在: {name}")
        del self._tools[name]

    def get(self, name: str) -> Tool:
        """按名称获取工具。"""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            raise ToolException(f"工具不存在: {name}. 可用工具: [{available}]")
        return tool

    def has(self, name: str) -> bool:
        """判断工具是否已注册。"""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """返回已注册工具列表。"""
        return list(self._tools.values())

    def names(self) -> list[str]:
        """返回已注册工具名称。"""
        return sorted(self._tools.keys())

    def export_schemas(self) -> list[dict[str, Any]]:
        """导出所有工具的 schema 描述。"""
        return [self._tools[name].to_dict() for name in self.names()]

    def execute(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """按工具名执行工具。"""
        tool = self.get(name)
        try:
            return tool.execute(params)
        except ToolException as exc:
            raise ToolException(f"工具 {name} 执行失败: {exc}") from exc

    def __len__(self) -> int:
        return len(self._tools)
