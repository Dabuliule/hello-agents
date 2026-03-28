"""Agent基类"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

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
            config: Optional[Config] = None
    ):
        normalized_name = name.strip() if isinstance(name, str) else ""
        if not normalized_name:
            raise ValueError("name 不能为空")

        normalized_system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None

        self.name = normalized_name
        self.llm = llm
        self.system_prompt = normalized_system_prompt or None
        self.config = config or Config()
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

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return [deepcopy(message) for message in self._history]

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
