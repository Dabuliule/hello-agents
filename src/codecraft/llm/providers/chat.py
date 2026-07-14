from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from codecraft.core.errors import ModelProviderError
from codecraft.llm.base import (
    LLMProtocolError,
    LLMProviderError,
    ModelRequest,
)
from codecraft.llm.events import ModelEvent, ModelEventType
from codecraft.llm.messages import ModelMessage, ModelMessageType
from codecraft.llm.providers._client import OpenAIClientProvider
from codecraft.llm.providers._protocol import (
    get_field,
    parse_arguments,
    required_string,
    serialize_arguments,
    token_value,
)
from codecraft.schema.tool import ToolCall, ToolSpec


class ChatCompletionsProvider(OpenAIClientProvider):
    """将 OpenAI 风格 Chat Completions 适配为内部事件。

    Chat 流以首个 choice 的 ``finish_reason`` 作为协议终点。当前运行时只请求
    一个候选答案，因此多 choice 响应会被视为协议错误，而不会静默拼接。
    """

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": self._messages_to_chat(request.messages),
            "stream": True,
            "max_tokens": request.max_output_tokens,
        }
        tools = self._tools_to_chat(request.tools)
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self._client().chat.completions.create(**kwargs)
            if hasattr(response, "__aiter__"):
                async for event in self._events_from_chat_stream(response):
                    yield event
                return
            for event in self._events_from_chat_response(response):
                yield event
        except ModelProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc

    @staticmethod
    def _messages_to_chat(
        messages: tuple[ModelMessage, ...] | list[ModelMessage],
    ) -> list[dict[str, Any]]:
        """转换消息，并把同一回复中的连续工具调用合并为一个 assistant 项。"""
        items: list[dict[str, Any]] = []
        pending_calls: list[ModelMessage] = []

        def flush_calls() -> None:
            if not pending_calls:
                return
            content = None
            if (
                items
                and items[-1].get("role") == "assistant"
                and "tool_calls" not in items[-1]
            ):
                content = items.pop()["content"]
            items.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        _message_to_chat_tool_call(message) for message in pending_calls
                    ],
                }
            )
            pending_calls.clear()

        for message in messages:
            if message.type == ModelMessageType.TOOL_CALL:
                pending_calls.append(message)
                continue
            flush_calls()
            items.append(_message_to_chat_item(message))
        flush_calls()
        return items

    @staticmethod
    def _tools_to_chat(
        tools: tuple[ToolSpec, ...] | list[ToolSpec],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
            if tool.enabled
        ]

    async def _events_from_chat_stream(
        self,
        stream: Any,
    ) -> AsyncIterator[ModelEvent]:
        """聚合被 chunk 拆分的工具参数，并校验流的正式结束原因。"""
        tool_parts: dict[int, dict[str, Any]] = {}
        latest_usage: dict[str, int] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            usage = self._chat_usage(chunk)
            if usage:
                latest_usage = usage

            choices = list(get_field(chunk, "choices", []) or [])
            if len(choices) > 1:
                raise LLMProtocolError("multiple chat choices are not supported")
            for choice in choices:
                choice_index = get_field(choice, "index", 0)
                if choice_index != 0:
                    raise LLMProtocolError("multiple chat choices are not supported")
                reason = get_field(choice, "finish_reason")
                if reason is not None:
                    if finish_reason is not None and finish_reason != reason:
                        raise LLMProtocolError(
                            "chat finish_reason changed within a stream"
                        )
                    finish_reason = str(reason)

                delta = get_field(choice, "delta", {}) or {}
                content = get_field(delta, "content")
                if content is not None:
                    if not isinstance(content, str):
                        raise LLMProtocolError("chat content delta must be text")
                    if content:
                        yield ModelEvent(
                            type=ModelEventType.MESSAGE_DELTA,
                            payload={"text": content},
                        )

                for raw_call in get_field(delta, "tool_calls", []) or []:
                    index = get_field(raw_call, "index", 0)
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                    ):
                        raise LLMProtocolError("tool call index must be non-negative")
                    part = tool_parts.setdefault(
                        index,
                        {"call_id": None, "name": None, "arguments": []},
                    )
                    call_id = get_field(raw_call, "id")
                    if call_id is not None:
                        _set_tool_part(part, "call_id", call_id)
                    function = get_field(raw_call, "function", {}) or {}
                    name = get_field(function, "name")
                    if name is not None:
                        _set_tool_part(part, "name", name)
                    arguments = get_field(function, "arguments")
                    if arguments is not None:
                        if not isinstance(arguments, str):
                            raise LLMProtocolError(
                                "streamed tool arguments must be strings"
                            )
                        part["arguments"].append(arguments)

        calls = [
            ToolCall(
                call_id=required_string(part["call_id"], "call_id"),
                name=required_string(part["name"], "name"),
                arguments=parse_arguments("".join(part["arguments"])),
            )
            for _, part in sorted(tool_parts.items())
        ]
        self._validate_finish_reason(finish_reason, bool(calls))

        if latest_usage:
            yield ModelEvent(type=ModelEventType.TOKEN_COUNT, payload=latest_usage)
        for call in calls:
            yield ModelEvent(type=ModelEventType.TOOL_CALL, payload=call)
        yield ModelEvent(type=ModelEventType.COMPLETED)

    def _events_from_chat_response(self, response: Any) -> list[ModelEvent]:
        choices = list(get_field(response, "choices", []) or [])
        if len(choices) != 1:
            raise LLMProtocolError("chat response must contain exactly one choice")

        choice = choices[0]
        if get_field(choice, "index", 0) != 0:
            raise LLMProtocolError("multiple chat choices are not supported")
        message = get_field(choice, "message", {}) or {}
        calls = _chat_tool_calls(message)
        self._validate_finish_reason(get_field(choice, "finish_reason"), bool(calls))

        events: list[ModelEvent] = []
        content = get_field(message, "content")
        if content is not None:
            if not isinstance(content, str):
                raise LLMProtocolError("chat message content must be text")
            if content:
                events.append(
                    ModelEvent(
                        type=ModelEventType.MESSAGE_COMPLETED,
                        payload={"text": content},
                    )
                )
        usage = self._chat_usage(response)
        if usage:
            events.append(ModelEvent(type=ModelEventType.TOKEN_COUNT, payload=usage))
        events.extend(
            ModelEvent(type=ModelEventType.TOOL_CALL, payload=call) for call in calls
        )
        events.append(ModelEvent(type=ModelEventType.COMPLETED))
        return events

    @staticmethod
    def _validate_finish_reason(reason: Any, has_tool_calls: bool) -> None:
        if reason is None:
            raise LLMProtocolError("chat response ended without finish_reason")
        if reason == "length":
            raise LLMProtocolError("chat response stopped at the token limit")
        if reason == "content_filter":
            raise LLMProviderError("chat response was blocked by the content filter")
        if reason not in {"stop", "tool_calls"}:
            raise LLMProtocolError(f"unsupported chat finish_reason: {reason}")
        if has_tool_calls != (reason == "tool_calls"):
            raise LLMProtocolError(
                "chat finish_reason does not match the returned tool calls"
            )

    @staticmethod
    def _chat_usage(response: Any) -> dict[str, int]:
        usage = get_field(response, "usage")
        if usage is None:
            return {}
        input_field = (
            "input_tokens"
            if get_field(usage, "input_tokens") is not None
            else "prompt_tokens"
        )
        output_field = (
            "output_tokens"
            if get_field(usage, "output_tokens") is not None
            else "completion_tokens"
        )
        input_tokens = token_value(usage, input_field)
        output_tokens = token_value(usage, output_field)
        prompt_details = get_field(usage, "prompt_tokens_details", {}) or {}
        completion_details = get_field(usage, "completion_tokens_details", {}) or {}
        total = get_field(usage, "total_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if total is None
            else token_value(usage, "total_tokens")
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": token_value(
                completion_details,
                "reasoning_tokens",
            ),
            "cached_input_tokens": token_value(prompt_details, "cached_tokens"),
            "total_tokens": total_tokens,
        }


def _message_to_chat_item(message: ModelMessage) -> dict[str, Any]:
    if message.type == ModelMessageType.TOOL_RESULT:
        assert message.tool_call_id is not None
        assert message.content is not None
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.type == ModelMessageType.TOOL_CALL:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_message_to_chat_tool_call(message)],
        }
    assert message.content is not None
    return {"role": message.role.value, "content": message.content}


def _message_to_chat_tool_call(message: ModelMessage) -> dict[str, Any]:
    assert message.tool_call_id is not None
    assert message.name is not None
    assert message.arguments is not None
    return {
        "id": message.tool_call_id,
        "type": "function",
        "function": {
            "name": message.name,
            "arguments": serialize_arguments(message.arguments),
        },
    }


def _chat_tool_calls(message: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    call_ids: set[str] = set()
    for raw_call in get_field(message, "tool_calls", []) or []:
        function = get_field(raw_call, "function", {}) or {}
        call = ToolCall(
            call_id=required_string(get_field(raw_call, "id"), "call_id"),
            name=required_string(get_field(function, "name"), "name"),
            arguments=parse_arguments(get_field(function, "arguments")),
        )
        if call.call_id in call_ids:
            raise LLMProtocolError(f"duplicate tool call id: {call.call_id}")
        call_ids.add(call.call_id)
        calls.append(call)
    return calls


def _set_tool_part(part: dict[str, Any], field: str, value: Any) -> None:
    previous = part[field]
    if previous is not None and previous != value:
        raise LLMProtocolError(f"streamed tool call changed its {field}")
    part[field] = value
