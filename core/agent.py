"""Agent基类"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Iterable, Optional

from tools.base import Tool
from tools.registry import ConflictPolicy, ToolRegistry
from .config import Config
from .llm import HelloAgentsLLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
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
        self.config = config or Config()
        self.tool_registry = tool_registry or ToolRegistry(tools)
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message) -> None:
        """添加消息到历史记录"""
        self._history.append(message)
        max_history = self.config.max_history_length
        if len(self._history) > max_history:
            # 只保留最新 N 条，避免历史无限增长
            self._history = self._history[-max_history:]

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
        """构造简洁工具说明，帮助模型知道可用工具。"""
        rows = [f"- {tool.name}: {tool.description}" for tool in self.tool_registry.list_tools()]
        return (
                "你可以建议用户使用以下工具（用户可通过 /tool 调用）：\n"
                + "\n".join(rows)
                + "\n/tool 用法: /tool <工具名> <JSON参数>"
        )

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return [deepcopy(message) for message in self._history]

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
