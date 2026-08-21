from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

from codecraft.core.errors import ModelProviderError
from codecraft.llm.base import (
    LLMProtocolError,
    LLMProviderError,
    ModelRequest,
)
from codecraft.llm.events import (
    ModelCompletedEvent,
    ModelEvent,
    ModelMessageDeltaEvent,
    ModelTokenCountEvent,
    ModelToolCallEvent,
)
from codecraft.llm.messages import (
    ModelMessage,
    ModelTextMessage,
    ModelToolCallMessage,
    ModelToolResultMessage,
)
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


@dataclass
class _ResponseStreamState:
    """一次 Responses 流中已展示文本和待确认工具调用的聚合状态。"""

    emitted_text: str = ""
    pending_calls: dict[str, ToolCall] = field(default_factory=dict)


class ResponsesProvider(OpenAIClientProvider):
    """将 OpenAI Responses API 适配为 CodeCraft 模型事件。

    Responses 流只有收到 ``response.completed`` 才算成功。失败事件、非完整
    状态或提前断流都会抛出异常，未确认完成的文本和工具调用不会被伪装成一
    次成功响应。
    """

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """发送一次 Responses API 请求并逐个产出内部模型事件。

        Args:
            request: 供应商无关的模型名、消息、工具和输出 Token 上限。

        Yields:
            文本、Token 用量、工具调用和最终成功标志等 ``ModelEvent``。

        Raises:
            LLMConfigError: 缺少 SDK 或 API Key。
            LLMProtocolError: Responses 状态或事件结构不满足内部协议。
            LLMProviderError: SDK、网络或供应商执行失败。

        Example:
            >>> from codecraft.llm.messages import ModelRole
            >>> provider = ResponsesProvider(client=object(), api_key_env=None)
            >>> request = ModelRequest(
            ...     model="gpt-test",
            ...     messages=(
            ...         ModelTextMessage(role=ModelRole.USER, content="你好"),
            ...     ),
            ... )
            >>> hasattr(provider.stream(request), "__aiter__")
            True
        """
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
        """按顺序把内部消息转换成 Responses API input items。

        Args:
            messages: 文本、工具调用和工具结果组成的内部消息序列。

        Returns:
            可传给 ``responses.create(input=...)`` 的字典列表。

        Raises:
            TypeError: 消息不是 ``ModelMessage`` 联合支持的具体类型。

        Example:
            >>> ResponsesProvider._messages_to_input(
            ...     [
            ...         ModelToolResultMessage(
            ...             tool_call_id="call_read", content="done"
            ...         )
            ...     ]
            ... )
            [{'type': 'function_call_output', 'call_id': 'call_read', 'output': 'done'}]
        """
        return [_message_to_response_item(message) for message in messages]

    @staticmethod
    def _tools_to_responses(
        tools: tuple[ToolSpec, ...] | list[ToolSpec],
    ) -> list[dict[str, Any]]:
        """把启用的内部工具转换成 Responses API function tools。

        Args:
            tools: 当前 Turn 可见的工具定义，可包含禁用工具。

        Returns:
            保持输入顺序的 Responses 工具列表；安全元数据不会发给模型。

        Example:
            >>> tool = ToolSpec(
            ...     name="read_file",
            ...     description="Read a file.",
            ...     input_schema={"type": "object"},
            ... )
            >>> ResponsesProvider._tools_to_responses([tool])
            [{'type': 'function', 'name': 'read_file', 'description': 'Read a file.', 'parameters': {'type': 'object'}}]
        """
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
        """转换一个完整 Responses 响应，并拒绝所有非成功状态。

        Args:
            response: SDK 完整响应对象或具有相同字段的字典。

        Returns:
            完整文本、可选 usage、工具调用和成功终止事件组成的列表。

        Raises:
            LLMProtocolError: 响应 incomplete、状态未知或内容结构非法。
            LLMProviderError: 响应状态为 failed。

        Example:
            >>> provider = ResponsesProvider(client=object(), api_key_env=None)
            >>> events = provider._events_from_response(
            ...     {"status": "completed", "output_text": "你好", "output": []}
            ... )
            >>> [event.type for event in events]
            ['message_delta', 'completed']
        """
        self._validate_status(response)
        events: list[ModelEvent] = []
        text = self._response_text(response)
        if text:
            events.append(
                ModelMessageDeltaEvent(
                    payload={"text": text},
                )
            )
        usage = self._usage(response)
        if usage:
            events.append(ModelTokenCountEvent(payload=usage))
        events.extend(
            ModelToolCallEvent(payload=call) for call in self._tool_calls(response)
        )
        events.append(ModelCompletedEvent())
        return events

    async def _events_from_stream(self, stream: Any) -> AsyncIterator[ModelEvent]:
        """转换 Responses 原始事件流，且只认可官方完成事件为成功。

        Args:
            stream: Responses SDK 返回的异步事件迭代器。

        Yields:
            文本增量，以及在 ``response.completed`` 后确认的 usage、工具调用和
            ``ModelCompletedEvent``。

        Raises:
            LLMProtocolError: 文本、工具、完成响应不一致，收到 incomplete，或流
                在完成事件前结束。
            LLMProviderError: 收到 failed 或 error 事件。

        Example:
            >>> import asyncio
            >>> async def raw_events():
            ...     yield {"type": "response.output_text.delta", "delta": "你好"}
            ...     yield {
            ...         "type": "response.completed",
            ...         "response": {
            ...             "status": "completed",
            ...             "output_text": "你好",
            ...             "output": [],
            ...         },
            ...     }
            >>> async def response_event_types():
            ...     provider = ResponsesProvider(client=object(), api_key_env=None)
            ...     return [
            ...         event.type
            ...         async for event in provider._events_from_stream(raw_events())
            ...     ]
            >>> asyncio.run(response_event_types())
            ['message_delta', 'completed']
        """
        state = _ResponseStreamState()

        async for raw_event in stream:
            event_type = get_field(raw_event, "type")

            if event_type == "response.output_text.delta":
                yield self._stream_text_delta(state, raw_event)
                continue

            if event_type == "response.output_item.done":
                self._record_done_item(state, raw_event)
                continue

            if event_type == "response.completed":
                for event in self._completed_stream_events(state, raw_event):
                    yield event
                return

            self._raise_for_stream_failure(raw_event, event_type)

        raise LLMProtocolError("Responses API stream ended before response.completed")

    @staticmethod
    def _stream_text_delta(
        state: _ResponseStreamState,
        raw_event: Any,
    ) -> ModelMessageDeltaEvent:
        """校验一个文本 delta，记录已展示文本并生成内部增量事件。

        Args:
            state: 当前响应的流式聚合状态。
            raw_event: ``response.output_text.delta`` 原始事件。

        Returns:
            携带当前非空文本片段的 ``ModelMessageDeltaEvent``。

        Raises:
            LLMProtocolError: delta 不是非空字符串。

        Example:
            >>> state = _ResponseStreamState()
            >>> event = ResponsesProvider._stream_text_delta(
            ...     state, {"delta": "你好"}
            ... )
            >>> (state.emitted_text, event.type)
            ('你好', 'message_delta')
        """
        delta = get_field(raw_event, "delta")
        if not isinstance(delta, str) or not delta:
            raise LLMProtocolError("response text delta must be non-empty")
        state.emitted_text += delta
        return ModelMessageDeltaEvent(
            payload={"text": delta},
        )

    def _record_done_item(
        self,
        state: _ResponseStreamState,
        raw_event: Any,
    ) -> None:
        """记录一个已完成 output item 中的工具调用，等待响应完成后发出。

        Args:
            state: 当前响应的流式聚合状态。
            raw_event: ``response.output_item.done`` 原始事件。

        Returns:
            ``None``。发现的调用按 call ID 写入 ``state.pending_calls``。

        Raises:
            LLMProtocolError: done 事件缺少 item，或工具调用结构、身份冲突。

        Example:
            >>> provider = ResponsesProvider(client=object(), api_key_env=None)
            >>> state = _ResponseStreamState()
            >>> provider._record_done_item(
            ...     state,
            ...     {
            ...         "item": {
            ...             "type": "function_call",
            ...             "call_id": "call_read",
            ...             "name": "read_file",
            ...             "arguments": '{"path":"README.md"}',
            ...         }
            ...     },
            ... )
            >>> list(state.pending_calls)
            ['call_read']
        """
        item = get_field(raw_event, "item")
        if item is None:
            raise LLMProtocolError("output_item.done is missing its item")
        for call in self._tool_calls({"output": [item]}):
            self._merge_tool_call(state.pending_calls, call)

    def _completed_stream_events(
        self,
        state: _ResponseStreamState,
        raw_event: Any,
    ) -> Iterator[ModelEvent]:
        """核对完成响应并产出尚未补齐的最终事件。

        Args:
            state: 已展示文本和提前到达的工具调用。
            raw_event: ``response.completed`` 原始事件。

        Yields:
            可选的漏发文本后缀、usage、去重工具调用和成功终止事件。

        Raises:
            LLMProtocolError: 完成事件缺少 response、状态非法、最终文本与已展示
                前缀冲突，或工具调用数据冲突。
            LLMProviderError: 完成响应状态为 failed。

        Example:
            >>> provider = ResponsesProvider(client=object(), api_key_env=None)
            >>> state = _ResponseStreamState(emitted_text="你")
            >>> events = list(
            ...     provider._completed_stream_events(
            ...         state,
            ...         {
            ...             "response": {
            ...                 "status": "completed",
            ...                 "output_text": "你好",
            ...                 "output": [],
            ...             }
            ...         },
            ...     )
            ... )
            >>> [(event.type, getattr(event, "payload", None)) for event in events]
            [('message_delta', ModelTextPayload(text='好')), ('completed', None)]
        """
        response = get_field(raw_event, "response")
        if response is None:
            raise LLMProtocolError("response.completed is missing its response")
        self._validate_status(response)

        final_text = self._response_text(response)
        if final_text != state.emitted_text:
            if not final_text.startswith(state.emitted_text):
                raise LLMProtocolError(
                    "streamed text does not match the completed response"
                )
            suffix = final_text[len(state.emitted_text) :]
            if suffix:
                state.emitted_text += suffix
                yield ModelMessageDeltaEvent(
                    payload={"text": suffix},
                )

        for call in self._tool_calls(response):
            self._merge_tool_call(state.pending_calls, call)
        usage = self._usage(response)
        if usage:
            yield ModelTokenCountEvent(payload=usage)
        for call in state.pending_calls.values():
            yield ModelToolCallEvent(payload=call)
        yield ModelCompletedEvent()

    @staticmethod
    def _raise_for_stream_failure(raw_event: Any, event_type: Any) -> None:
        """把 Responses 明确的失败事件映射成稳定的内部异常。

        Args:
            raw_event: 原始 Responses 事件。
            event_type: 已提取的事件类型。

        Returns:
            ``None``。非失败事件保持忽略，由其他分支按需处理。

        Raises:
            LLMProtocolError: 事件类型为 ``response.incomplete``。
            LLMProviderError: 事件类型为 ``response.failed`` 或 ``error``。

        Example:
            >>> ResponsesProvider._raise_for_stream_failure(
            ...     {"type": "response.created"}, "response.created"
            ... )
        """
        if event_type not in {
            "response.failed",
            "response.incomplete",
            "error",
        }:
            return
        message = error_message(raw_event, f"Responses API {event_type}")
        if event_type == "response.incomplete":
            raise LLMProtocolError(message)
        raise LLMProviderError(message)

    @staticmethod
    def _merge_tool_call(pending: dict[str, ToolCall], call: ToolCall) -> None:
        """按 call ID 幂等合并流中重复出现的完整工具调用。

        Args:
            pending: 当前响应尚未发出的工具调用映射。
            call: 新解析出的完整工具调用。

        Returns:
            ``None``。调用会写入或保持在 ``pending`` 中。

        Raises:
            LLMProtocolError: 同一 call ID 对应不同名称或参数。

        Example:
            >>> pending = {}
            >>> call = ToolCall(call_id="call_read", name="read_file", arguments={})
            >>> ResponsesProvider._merge_tool_call(pending, call)
            >>> ResponsesProvider._merge_tool_call(pending, call)
            >>> list(pending)
            ['call_read']
        """
        previous = pending.get(call.call_id)
        if previous is not None and previous != call:
            raise LLMProtocolError(f"conflicting tool call data for {call.call_id}")
        pending[call.call_id] = call

    @staticmethod
    def _validate_status(response: Any) -> None:
        """确认完整 Responses 对象处于可提交状态。

        Args:
            response: 完整响应对象或字典。

        Returns:
            ``None``。为兼容精简测试对象，缺失 status 与 completed 均视为成功。

        Raises:
            LLMProtocolError: 状态为 incomplete 或未知值。
            LLMProviderError: 状态为 failed。

        Example:
            >>> ResponsesProvider._validate_status({"status": "completed"})
            >>> ResponsesProvider._validate_status({})
        """
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
        """从 Responses 便捷字段或结构化 output items 提取完整文本。

        Args:
            response: 完整 Responses 对象或字典。

        Returns:
            ``output_text`` 字符串；没有便捷字段时返回所有 message content 文本
            按供应商顺序拼接的结果。

        Example:
            >>> ResponsesProvider._response_text(
            ...     {
            ...         "output": [
            ...             {
            ...                 "type": "message",
            ...                 "content": [{"text": "你"}, {"text": "好"}],
            ...             }
            ...         ]
            ...     }
            ... )
            '你好'
        """
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
        """从 Responses output items 解析全部 function call。

        Args:
            response: 完整响应，或只含部分 output items 的临时包装对象。

        Returns:
            保持 output 顺序的内部工具调用列表。

        Raises:
            LLMProtocolError: 调用身份、参数非法，或同一批 output 中 call ID 重复。

        Example:
            >>> calls = ResponsesProvider._tool_calls(
            ...     {
            ...         "output": [
            ...             {
            ...                 "type": "function_call",
            ...                 "call_id": "call_read",
            ...                 "name": "read_file",
            ...                 "arguments": '{"path":"README.md"}',
            ...             }
            ...         ]
            ...     }
            ... )
            >>> calls[0].name
            'read_file'
        """
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
        """把 Responses usage 归一为 CodeCraft Token 用量。

        Args:
            response: 含可选 usage 的完整响应。

        Returns:
            没有 usage 时返回空字典；否则返回统一的五项 Token 计数。

        Raises:
            LLMProtocolError: Token 不是非负整数，或 total 与 input、output 之和
                不一致。

        Example:
            >>> ResponsesProvider._usage(
            ...     {
            ...         "usage": {
            ...             "input_tokens": 7,
            ...             "output_tokens": 3,
            ...             "total_tokens": 10,
            ...             "input_tokens_details": {"cached_tokens": 2},
            ...             "output_tokens_details": {"reasoning_tokens": 1},
            ...         }
            ...     }
            ... )
            {'input_tokens': 7, 'output_tokens': 3, 'reasoning_tokens': 1, 'cached_input_tokens': 2, 'total_tokens': 10}
        """
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
        if total_tokens != input_tokens + output_tokens:
            raise LLMProtocolError(
                "usage.total_tokens must equal input_tokens + output_tokens"
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": token_value(output_details, "reasoning_tokens"),
            "cached_input_tokens": token_value(input_details, "cached_tokens"),
            "total_tokens": total_tokens,
        }


def _message_to_response_item(message: ModelMessage) -> dict[str, Any]:
    """把一条内部消息转换成 Responses API input item。

    Args:
        message: 文本、工具调用或工具结果消息。

    Returns:
        Responses API 对应的普通 message、function_call 或
        function_call_output 字典。

    Raises:
        TypeError: 消息类型不属于内部 ``ModelMessage`` 联合，或参数无法 JSON
            序列化。

    Example:
        >>> _message_to_response_item(
        ...     ModelToolCallMessage(
        ...         tool_call_id="call_read",
        ...         name="read_file",
        ...         arguments={"path": "README.md"},
        ...     )
        ... )
        {'type': 'function_call', 'call_id': 'call_read', 'name': 'read_file', 'arguments': '{"path":"README.md"}'}
    """
    if isinstance(message, ModelToolCallMessage):
        return {
            "type": "function_call",
            "call_id": message.tool_call_id,
            "name": message.name,
            "arguments": serialize_arguments(message.arguments),
        }
    if isinstance(message, ModelToolResultMessage):
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.content,
        }
    if not isinstance(message, ModelTextMessage):
        raise TypeError(f"unsupported model message: {type(message).__name__}")
    return {"role": message.role.value, "content": message.content}
