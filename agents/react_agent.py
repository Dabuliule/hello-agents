"""ReAct 风格 Agent，支持工具调用。"""

import json
import re
from typing import Any, Iterable, Iterator

from core.agent import Agent
from core.exceptions import AgentException, HelloAgentsException
from core.message import Message
from tools.base import Tool
from tools.registry import ToolRegistry


class ReActAgent(Agent):
    """遵循 Thought/Action/Observation 循环的 Agent。"""

    def __init__(
            self,
            name: str,
            llm,
            system_prompt: str | None = None,
            config=None,
            tools: Iterable[Tool] | None = None,
            tool_registry: ToolRegistry | None = None,
            max_iterations: int = 8,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tools=tools,
            tool_registry=tool_registry,
        )
        self.max_iterations = max_iterations

    def run(self, input_text: str, **kwargs: Any) -> str:
        """非流式执行，返回最终答案。"""
        user_text = self._normalize_input(input_text)
        self.add_message(Message(role="user", content=user_text))
        return self._reasoning_loop(**kwargs)

    def run_stream(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        """流式执行，产出模型输出和观察结果。"""
        user_text = self._normalize_input(input_text)
        self.add_message(Message(role="user", content=user_text))
        yield from self._reasoning_loop_stream(**kwargs)

    def _reasoning_loop(self, **kwargs: Any) -> str:
        for _ in range(self.max_iterations):
            response = self._invoke_llm(**kwargs)
            self.add_message(Message(role="assistant", content=response))

            final_answer = self._extract_final_answer(response)
            if final_answer is not None:
                return final_answer

            tool_name, tool_params = self._extract_action(response)
            if not tool_name:
                raise AgentException("ReActAgent 未提供可执行的动作。")

            self.call_tool(tool_name, tool_params)

        raise AgentException("ReActAgent 超过最大迭代次数仍未给出最终答案。")

    def _reasoning_loop_stream(self, **kwargs: Any) -> Iterator[str]:
        for _ in range(self.max_iterations):
            chunks: list[str] = []
            try:
                for chunk in self.llm.stream_invoke(self._build_messages(), **kwargs):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    yield chunk
            except HelloAgentsException as exc:
                raise AgentException(f"ReActAgent 流式调用失败: {exc}") from exc

            response = "".join(chunks).strip()
            if not response:
                continue

            self.add_message(Message(role="assistant", content=response))

            final_answer = self._extract_final_answer(response)
            if final_answer is not None:
                return

            tool_name, tool_params = self._extract_action(response)
            if not tool_name:
                raise AgentException("ReActAgent 未提供可执行的动作。")

            result = self.call_tool(tool_name, tool_params)
            yield f"\n观察结果: {result}\n"

        raise AgentException("ReActAgent 超过最大迭代次数仍未给出最终答案。")

    def _invoke_llm(self, **kwargs: Any) -> str:
        try:
            return self.llm.invoke(self._build_messages(), **kwargs) or ""
        except HelloAgentsException as exc:
            raise AgentException(f"ReActAgent 调用 LLM 失败: {exc}") from exc

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_blocks: list[str] = []

        if self.system_prompt:
            system_blocks.append(self.system_prompt)

        system_blocks.append(self._react_format_hint())

        tool_hint = self._build_tool_hint()
        if tool_hint:
            system_blocks.append(tool_hint)

        if system_blocks:
            messages.append({"role": "system", "content": "\n\n".join(system_blocks)})

        # TODO: tool 角色不适配当前调用方式，转为 assistant 的 Observation
        for message in self.get_history():
            if message.role == "tool":
                messages.append({"role": "assistant", "content": f"Observation: {message.content}"})
            else:
                messages.append(message.to_dict())

        return messages

    @staticmethod
    def _react_format_hint() -> str:
        return (
            "严格遵守以下格式，除非给出 Final Answer，否则必须同时输出 Thought、Action 和 Action Input:\n"
            "Thought: ...\n"
            "Action: <工具名>\n"
            "Action Input: {\"参数\": \"值\"}\n\n"
            "当你完成时:\n"
            "Final Answer: ..."
        )

    @staticmethod
    def _normalize_input(input_text: str) -> str:
        text = input_text.strip() if isinstance(input_text, str) else ""
        if not text:
            raise AgentException("input_text 不能为空")
        return text

    @staticmethod
    def _extract_final_answer(response: str) -> str | None:
        match = re.search(r"Final Answer:\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    @staticmethod
    def _extract_action(response: str) -> tuple[str | None, dict[str, Any]]:
        if re.search(r"^\s*Observation:\s*", response, re.IGNORECASE | re.MULTILINE):
            raise AgentException("模型输出包含 Observation，违反格式要求。")

        action_match = re.search(r"Action:\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
        if not action_match:
            return None, {}

        action_line = action_match.group(1).strip()
        if action_line.lower().startswith("final answer"):
            return None, {}

        tool_name = action_line.split()[0].strip()
        input_match = re.search(r"Action Input:\s*(\{.+})", response, re.IGNORECASE | re.DOTALL)
        raw_input = input_match.group(1).strip() if input_match else "{}"

        try:
            params = json.loads(raw_input)
        except json.JSONDecodeError as exc:
            raise AgentException(f"动作输入的 JSON 格式无效: {exc}") from exc

        if not isinstance(params, dict):
            raise AgentException("动作输入必须是 JSON 对象。")

        return tool_name, params
