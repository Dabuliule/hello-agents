"""工具图，支持多工具编排。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from core.exceptions import ToolException
from .registry import ToolRegistry


class ToolGraphException(ToolException):
    """工具图异常。"""


@dataclass
class ToolGraphNode:
    """工具图节点。"""

    node_id: str
    tool_name: str
    input_map: Dict[str, str] = field(default_factory=dict)
    static_params: Dict[str, Any] = field(default_factory=dict)
    output_key: str | None = None


class ToolGraph:
    """基于有向无环图的工具图。"""

    def __init__(self, registry: ToolRegistry, name: str = "tool_graph"):
        self.name = name
        self.registry = registry
        self._nodes: dict[str, ToolGraphNode] = {}
        self._edges: dict[str, set[str]] = {}

    def add_node(
            self,
            node_id: str,
            tool_name: str,
            input_map: dict[str, str] | None = None,
            static_params: dict[str, Any] | None = None,
            output_key: str | None = None,
    ) -> None:
        """添加节点。"""
        if node_id in self._nodes:
            raise ToolGraphException(f"节点已存在: {node_id}")
        if not self.registry.has(tool_name):
            raise ToolGraphException(f"工具不存在: {tool_name}")

        self._nodes[node_id] = ToolGraphNode(
            node_id=node_id,
            tool_name=tool_name,
            input_map=input_map or {},
            static_params=static_params or {},
            output_key=output_key,
        )
        self._edges.setdefault(node_id, set())

    def add_edge(self, from_node: str, to_node: str) -> None:
        """添加依赖边。"""
        if from_node not in self._nodes or to_node not in self._nodes:
            raise ToolGraphException("边的节点不存在")
        self._edges.setdefault(from_node, set()).add(to_node)

    def run(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行工具链，返回上下文。"""
        context: dict[str, Any] = dict(inputs or {})
        order = self._topological_sort()

        for node_id in order:
            node = self._nodes[node_id]
            params = dict(node.static_params)
            for param_name, key in node.input_map.items():
                if key not in context:
                    raise ToolGraphException(f"缺少输入: {key}")
                params[param_name] = context[key]

            result = self.registry.execute(node.tool_name, params)
            output_key = node.output_key or node_id
            context[output_key] = result

        return context

    def _topological_sort(self) -> list[str]:
        """拓扑排序，保证依赖顺序。"""
        in_degree: dict[str, int] = {node_id: 0 for node_id in self._nodes}
        for from_node, to_nodes in self._edges.items():
            for to_node in to_nodes:
                in_degree[to_node] += 1

        ready = [node_id for node_id, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while ready:
            node_id = ready.pop()
            order.append(node_id)
            for to_node in self._edges.get(node_id, set()):
                in_degree[to_node] -= 1
                if in_degree[to_node] == 0:
                    ready.append(to_node)

        if len(order) != len(self._nodes):
            raise ToolGraphException("工具图存在环")
        return order

    def nodes(self) -> Iterable[ToolGraphNode]:
        """返回所有节点。"""
        return self._nodes.values()
