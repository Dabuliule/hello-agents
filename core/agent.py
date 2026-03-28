"""Agent基类"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Iterable, Optional

from tools.base import Tool
from tools.registry import ConflictPolicy, ToolRegistry
from .llm import HelloAgentsLLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            history_limit: int = 100,
            tools: Iterable[Tool] | None = None,
            tool_registry: ToolRegistry | None = None,
    ):
        normalized_name = name.strip() if isinstance(name, str) else ""
        if not normalized_name:
            raise ValueError("name 不能为空")

        normalized_system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None

        self.name = normalized_name
        self.llm = llm
        self.system_prompt = normalized_system_prompt or None
        if history_limit < 1:
            raise ValueError("history_limit 必须大于 0")
        self.history_limit = history_limit
        self.tool_registry = tool_registry or ToolRegistry(tools)
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message) -> None:
        """添加消息到历史记录"""
        self._history.append(message)
        if len(self._history) > self.history_limit:
            # 只保留最新 N 条，避免历史无限增长
            self._history = self._history[-self.history_limit:]

    def clear_history(self) -> None:
        """清空历史记录"""
        self._history.clear()

    def register_tool(self, tool: Tool, conflict: ConflictPolicy = "error") -> None:
        """注册单个工具。"""
        self.tool_registry.register(tool, conflict=conflict)

    def available_tools(self) -> list[str]:
        """返回可用工具名称。"""
        return self.tool_registry.names()

    def call_tool(self, name: str, params: dict[str, Any] | None = None) -> Any:
        """执行工具并写入工具消息历史。"""
        result = self.tool_registry.execute(name, params)
        self.add_message(
            Message(
                role="tool",
                content=f"{name}: {result}",
                metadata={"tool": name, "params": params or {}},
            )
        )
        return result

    def _build_tool_hint(self) -> str:
        """构造工具说明，包含名称、描述和参数 schema。"""
        tools = self.tool_registry.list_tools()
        if not tools:
            return "当前没有可用工具。"
        rows = [
            f"- {tool.name}: {tool.description}\n  schema: {tool.params_model.model_json_schema()}"
            for tool in tools
        ]
        return "可用工具如下：\n" + "\n".join(rows)

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return [deepcopy(message) for message in self._history]

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
