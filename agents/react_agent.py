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

        if system_blocks:
            messages.append({"role": "system", "content": "\n\n".join(system_blocks)})

        # TODO: tool 角色不适配当前调用方式，转为 assistant 的 Observation
        for message in self.get_history():
            if message.role == "tool":
                messages.append({"role": "assistant", "content": f"Observation: {message.content}"})
            else:
                messages.append(message.to_dict())

        return messages

    def _react_format_hint(self) -> str:
        tool_section = f"\n\n【可用工具】\n{self._build_tool_hint()}\n"
        return (
            "你必须严格按照 ReAct 规范进行推理与工具调用，所有输出必须遵循以下规则：\n\n"

            "【基本规则】\n"
            "1. 每一轮输出只能包含以下两种结构之一：\n"
            "   (A) Thought + Action + Action Input\n"
            "   (B) Final Answer\n"
            "2. 除 Final Answer 外，必须同时包含 Thought、Action 和 Action Input，缺一不可\n"
            "3. 严禁输出多余字段（如 Observation、Explanation 等）\n"
            "4. 严禁在同一轮中调用多个工具（只能有一个 Action）\n"
            f"{tool_section}\n"
            "【格式要求】\n"
            "严格按照以下格式输出（大小写敏感，字段名不可更改）：\n\n"
            "Thought: <你的思考过程，简洁但清晰>\n"
            "Action: <工具名称，必须是已提供的工具之一>\n"
            "Action Input: <JSON字符串，必须是合法JSON>\n\n"

            "示例：\n"
            "Thought: 需要查询用户信息\n"
            "Action: get_user_info\n"
            "Action Input: {\"user_id\": \"123\"}\n\n"

            "【Final Answer 规则】\n"
            "当你已经获得足够信息，可以直接回答用户问题时，必须输出：\n\n"
            "Final Answer: <最终答案>\n\n"

            "注意：\n"
            "- 输出 Final Answer 时，不得再包含 Thought / Action / Action Input\n"
            "- Final Answer 应直接回答用户问题，不要包含额外格式\n\n"

            "【JSON 规范】\n"
            "- Action Input 必须是严格合法的 JSON（使用双引号）\n"
            "- 不允许使用单引号、注释或 trailing comma\n"
            "- 参数必须与工具定义完全匹配\n\n"

            "【错误处理】\n"
            "- 如果你不确定使用哪个工具，请先在 Thought 中分析，再选择最合适的工具\n"
            "- 如果上一步工具结果不正确，请重新思考并调用工具\n"
            "- 不要假造工具结果，必须依赖真实 Observation（系统返回）\n\n"

            "请严格遵守以上规则进行输出。"
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
