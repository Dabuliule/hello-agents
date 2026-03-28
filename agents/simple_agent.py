"""简单对话 Agent，实现流式和非流式调用"""

from typing import Any, Iterator

from core.agent import Agent
from core.exceptions import AgentException, HelloAgentsException
from core.message import Message


class SimpleAgent(Agent):
    """最小可用对话 Agent"""

    def run(self, input_text: str, **kwargs: Any) -> str:
        """非流式对话，返回完整回答"""
        user_text = self._normalize_input(input_text)
        self.add_message(Message(role="user", content=user_text))

        try:
            response = self.llm.invoke(self._build_messages(), **kwargs)
        except HelloAgentsException as exc:
            raise AgentException(f"SimpleAgent 非流式调用失败: {exc}") from exc
        except Exception as exc:
            raise AgentException(f"SimpleAgent 发生未知错误: {exc}") from exc

        assistant_text = (response or "").strip()
        if assistant_text:
            self.add_message(Message(role="assistant", content=assistant_text))
        return assistant_text

    def run_stream(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        """流式对话，逐段返回回答"""
        user_text = self._normalize_input(input_text)
        self.add_message(Message(role="user", content=user_text))

        chunks: list[str] = []
        try:
            for chunk in self.llm.stream_invoke(self._build_messages(), **kwargs):
                if not chunk:
                    continue
                chunks.append(chunk)
                yield chunk
        except HelloAgentsException as exc:
            raise AgentException(f"SimpleAgent 流式调用失败: {exc}") from exc
        except Exception as exc:
            raise AgentException(f"SimpleAgent 发生未知错误: {exc}") from exc

        assistant_text = "".join(chunks).strip()
        if assistant_text:
            self.add_message(Message(role="assistant", content=assistant_text))

    def _build_messages(self) -> list[dict[str, str]]:
        """将 system_prompt + history 转为 LLM 入参"""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(message.to_dict() for message in self.get_history())
        return messages

    @staticmethod
    def _normalize_input(input_text: str) -> str:
        """输入预处理，避免空消息"""
        text = input_text.strip() if isinstance(input_text, str) else ""
        if not text:
            raise AgentException("input_text 不能为空")
        return text
