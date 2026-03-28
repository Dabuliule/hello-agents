"""简单对话 Agent，实现流式和非流式调用"""

import json
from typing import Any, Iterable, Iterator

from core.agent import Agent
from core.exceptions import AgentException, HelloAgentsException, ToolException
from core.message import Message
from tools.base import Tool
from tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """最小可用对话 Agent"""

    def __init__(
            self,
            name: str,
            llm,
            system_prompt: str | None = None,
            history_limit: int = 100,
            tools: Iterable[Tool] | None = None,
            tool_registry: ToolRegistry | None = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            history_limit=history_limit,
            tools=tools,
            tool_registry=tool_registry,
        )

    def run(self, input_text: str, **kwargs: Any) -> str:
        """非流式对话，返回完整回答"""
        user_text = self._normalize_input(input_text)
        self.add_message(Message(role="user", content=user_text))

        tool_response = self._maybe_run_tool_command(user_text)
        if tool_response is not None:
            self.add_message(Message(role="assistant", content=tool_response))
            return tool_response

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

        tool_response = self._maybe_run_tool_command(user_text)
        if tool_response is not None:
            self.add_message(Message(role="assistant", content=tool_response))
            yield tool_response
            return

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
        system_blocks: list[str] = []
        if self.system_prompt:
            system_blocks.append(self.system_prompt)
        if len(self.tool_registry) > 0:
            system_blocks.append(self._build_tool_hint())
        if system_blocks:
            messages.append({"role": "system", "content": "\n\n".join(system_blocks)})
        messages.extend(message.to_dict() for message in self.get_history())
        return messages

    def _maybe_run_tool_command(self, user_text: str) -> str | None:
        """处理 /tool 命令，命令格式: /tool <name> <json>"""
        if not user_text.startswith("/tool"):
            return None

        parts = user_text.split(maxsplit=2)
        if len(parts) < 2:
            raise AgentException("工具命令格式错误，应为: /tool <工具名> <JSON参数>")

        tool_name = parts[1].strip()
        params: dict[str, Any] = {}
        if len(parts) == 3 and parts[2].strip():
            try:
                loaded = json.loads(parts[2])
            except json.JSONDecodeError as exc:
                raise AgentException(f"工具参数 JSON 解析失败: {exc}") from exc
            if not isinstance(loaded, dict):
                raise AgentException("工具参数必须是 JSON 对象")
            params = loaded

        try:
            result = self.call_tool(tool_name, params)
        except ToolException as exc:
            raise AgentException(f"工具调用失败: {exc}") from exc

        return f"工具 {tool_name} 执行结果: {result}"

    @staticmethod
    def _normalize_input(input_text: str) -> str:
        """输入预处理，避免空消息"""
        text = input_text.strip() if isinstance(input_text, str) else ""
        if not text:
            raise AgentException("input_text 不能为空")
        return text
