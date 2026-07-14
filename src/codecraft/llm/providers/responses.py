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
    error_message,
    get_field,
    parse_arguments,
    required_string,
    serialize_arguments,
    token_value,
)
from codecraft.schema.tool import ToolCall, ToolSpec


class ResponsesProvider(OpenAIClientProvider):
    """将 OpenAI Responses API 适配为 CodeCraft 模型事件。

    Responses 流只有收到 ``response.completed`` 才算成功。失败事件、非完整
    状态或提前断流都会抛出异常，未确认完成的文本和工具调用不会被伪装成一
    次成功响应。
    """

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": self._messages_to_input(request.messages),
            "stream": True,
            "store": False,
            "max_output_tokens": request.max_output_tokens,
        }
        tools = self._tools_to_responses(request.tools)
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self._client().responses.create(**kwargs)
            if hasattr(response, "__aiter__"):
                async for event in self._events_from_stream(response):
                    yield event
                return
            for event in self._events_from_response(response):
                yield event
        except ModelProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc

    @staticmethod
    def _messages_to_input(
        messages: tuple[ModelMessage, ...] | list[ModelMessage],
    ) -> list[dict[str, Any]]:
        return [_message_to_response_item(message) for message in messages]

    @staticmethod
    def _tools_to_responses(
        tools: tuple[ToolSpec, ...] | list[ToolSpec],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
            if tool.enabled
        ]

    def _events_from_response(self, response: Any) -> list[ModelEvent]:
        """转换完整响应，并拒绝 failed、incomplete 等非成功状态。"""
        self._validate_status(response)
        events: list[ModelEvent] = []
        text = self._response_text(response)
        if text:
            events.append(
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": text},
                )
            )
        usage = self._usage(response)
        if usage:
            events.append(ModelEvent(type=ModelEventType.TOKEN_COUNT, payload=usage))
        events.extend(
            ModelEvent(type=ModelEventType.TOOL_CALL, payload=call)
            for call in self._tool_calls(response)
        )
        events.append(ModelEvent(type=ModelEventType.COMPLETED))
        return events

    async def _events_from_stream(self, stream: Any) -> AsyncIterator[ModelEvent]:
        """转换原始事件流；只有官方完成事件能够生成 ``COMPLETED``。"""
        emitted_text = ""
        pending_calls: dict[str, ToolCall] = {}

        async for raw_event in stream:
            event_type = get_field(raw_event, "type")

            if event_type == "response.output_text.delta":
                delta = get_field(raw_event, "delta")
                if not isinstance(delta, str) or not delta:
                    raise LLMProtocolError("response text delta must be non-empty")
                emitted_text += delta
                yield ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": delta},
                )
                continue

            if event_type == "response.output_item.done":
                item = get_field(raw_event, "item")
                if item is None:
                    raise LLMProtocolError("output_item.done is missing its item")
                for call in self._tool_calls({"output": [item]}):
                    self._merge_tool_call(pending_calls, call)
                continue

            if event_type == "response.completed":
                response = get_field(raw_event, "response")
                if response is None:
                    raise LLMProtocolError("response.completed is missing its response")
                self._validate_status(response)

                final_text = self._response_text(response)
                if final_text != emitted_text:
                    if not final_text.startswith(emitted_text):
                        raise LLMProtocolError(
                            "streamed text does not match the completed response"
                        )
                    suffix = final_text[len(emitted_text) :]
                    if suffix:
                        emitted_text += suffix
                        yield ModelEvent(
                            type=ModelEventType.MESSAGE_DELTA,
                            payload={"text": suffix},
                        )

                for call in self._tool_calls(response):
                    self._merge_tool_call(pending_calls, call)
                usage = self._usage(response)
                if usage:
                    yield ModelEvent(
                        type=ModelEventType.TOKEN_COUNT,
                        payload=usage,
                    )
                for call in pending_calls.values():
                    yield ModelEvent(type=ModelEventType.TOOL_CALL, payload=call)
                yield ModelEvent(type=ModelEventType.COMPLETED)
                return

            if event_type in {
                "response.failed",
                "response.incomplete",
                "error",
            }:
                message = error_message(raw_event, f"Responses API {event_type}")
                if event_type == "response.incomplete":
                    raise LLMProtocolError(message)
                raise LLMProviderError(message)

        raise LLMProtocolError("Responses API stream ended before response.completed")

    @staticmethod
    def _merge_tool_call(pending: dict[str, ToolCall], call: ToolCall) -> None:
        previous = pending.get(call.call_id)
        if previous is not None and previous != call:
            raise LLMProtocolError(f"conflicting tool call data for {call.call_id}")
        pending[call.call_id] = call

    @staticmethod
    def _validate_status(response: Any) -> None:
        status = get_field(response, "status")
        if status is None or status == "completed":
            return
        if status == "failed":
            raise LLMProviderError(error_message(response, "Responses API failed"))
        if status == "incomplete":
            details = get_field(response, "incomplete_details")
            reason = get_field(details, "reason", "unknown reason")
            raise LLMProtocolError(f"Responses API returned incomplete: {reason}")
        raise LLMProtocolError(f"unexpected Responses API status: {status}")

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = get_field(response, "output_text")
        if isinstance(output_text, str):
            return output_text

        parts: list[str] = []
        for item in get_field(response, "output", []) or []:
            if get_field(item, "type") != "message":
                continue
            for content in get_field(item, "content", []) or []:
                text = get_field(content, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _tool_calls(response: Any) -> list[ToolCall]:
        calls: list[ToolCall] = []
        call_ids: set[str] = set()
        for item in get_field(response, "output", []) or []:
            if get_field(item, "type") != "function_call":
                continue
            call = ToolCall(
                call_id=required_string(
                    get_field(item, "call_id"),
                    "call_id",
                ),
                name=required_string(get_field(item, "name"), "name"),
                arguments=parse_arguments(get_field(item, "arguments")),
            )
            if call.call_id in call_ids:
                raise LLMProtocolError(f"duplicate tool call id: {call.call_id}")
            call_ids.add(call.call_id)
            calls.append(call)
        return calls

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = get_field(response, "usage")
        if usage is None:
            return {}
        input_tokens = token_value(usage, "input_tokens")
        output_tokens = token_value(usage, "output_tokens")
        input_details = get_field(usage, "input_tokens_details", {}) or {}
        output_details = get_field(usage, "output_tokens_details", {}) or {}
        total = get_field(usage, "total_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if total is None
            else token_value(usage, "total_tokens")
        )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": token_value(output_details, "reasoning_tokens"),
            "cached_input_tokens": token_value(input_details, "cached_tokens"),
            "total_tokens": total_tokens,
        }


def _message_to_response_item(message: ModelMessage) -> dict[str, Any]:
    if message.type == ModelMessageType.TOOL_CALL:
        assert message.tool_call_id is not None
        assert message.name is not None
        assert message.arguments is not None
        return {
            "type": "function_call",
            "call_id": message.tool_call_id,
            "name": message.name,
            "arguments": serialize_arguments(message.arguments),
        }
    if message.type == ModelMessageType.TOOL_RESULT:
        assert message.tool_call_id is not None
        assert message.content is not None
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        }
    assert message.content is not None
    return {"role": message.role.value, "content": message.content}
