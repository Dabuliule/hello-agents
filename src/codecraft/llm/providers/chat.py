from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal

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
        """发送一次 Chat Completions 请求并逐个产出内部模型事件。

        Args:
            request: 不可变的模型请求，包含模型名、历史消息、可用工具和最大输出
                Token 数。

        Yields:
            由上游响应转换得到的 ``ModelEvent``。文本流会先产生若干
            ``ModelMessageDeltaEvent``，随后可产生 Token 用量和工具调用，最后
            一定以 ``ModelCompletedEvent`` 表示成功结束。

        Raises:
            LLMConfigError: 创建客户端时缺少 API Key 或 OpenAI SDK。
            LLMProtocolError: 上游响应缺少结束原因，或返回了不支持的数据结构。
            LLMProviderError: SDK 请求、网络传输或内容过滤失败。

        Example:
            ``stream`` 是异步迭代器，需要用 ``async for`` 消费：

            >>> from codecraft.llm.base import ModelRequest
            >>> from codecraft.llm.messages import ModelRole, ModelTextMessage
            >>> from codecraft.llm.providers.qwen import QwenProvider
            >>> provider = QwenProvider(api_key="test-key")
            >>> request = ModelRequest(
            ...     model="qwen-plus",
            ...     messages=(
            ...         ModelTextMessage(role=ModelRole.USER, content="你好"),
            ...     ),
            ... )
            >>> events = provider.stream(request)
            >>> hasattr(events, "__aiter__")
            True
        """
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
        """把内部消息转换成 Chat Completions 的 ``messages`` 数组。

        同一模型回复中的连续工具调用在内部被保存为多个独立消息；Chat
        Completions 协议则要求它们位于同一个 assistant 消息的 ``tool_calls``
        数组中。本方法会缓冲并合并这些调用，同时保留紧邻它们的 assistant
        文本内容。

        Args:
            messages: 按对话顺序排列的内部文本、工具调用和工具结果消息。

        Returns:
            可直接传给 ``chat.completions.create(messages=...)`` 的字典列表。

        Raises:
            TypeError: 消息不是 ``ModelMessage`` 支持的具体类型。

        Example:
            两个连续工具调用会合并到同一个 assistant 消息中：

            >>> calls = [
            ...     ModelToolCallMessage(
            ...         name="read_file",
            ...         tool_call_id="call_a",
            ...         arguments={"path": "a.py"},
            ...     ),
            ...     ModelToolCallMessage(
            ...         name="read_file",
            ...         tool_call_id="call_b",
            ...         arguments={"path": "b.py"},
            ...     ),
            ... ]
            >>> converted = ChatCompletionsProvider._messages_to_chat(calls)
            >>> len(converted)
            1
            >>> [call["id"] for call in converted[0]["tool_calls"]]
            ['call_a', 'call_b']
        """
        items: list[dict[str, Any]] = []
        pending_calls: list[ModelToolCallMessage] = []

        def flush_calls() -> None:
            """把已缓冲的工具调用写入 ``items``，并清空缓冲区。"""
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
            if isinstance(message, ModelToolCallMessage):
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
        """把启用的内部工具定义转换成 Chat Completions function tools。

        ``effects`` 和 ``requires_approval`` 等安全元数据只供 Runtime 决策，不会
        暴露给模型。模型只需要知道工具名称、用途和参数 JSON Schema。

        Args:
            tools: 当前 Turn 可见的内部工具定义，允许包含已禁用工具。

        Returns:
            保持输入顺序的 Chat Completions 工具列表；禁用工具会被过滤。

        Example:
            >>> tool = ToolSpec(
            ...     name="read_file",
            ...     description="Read one workspace file.",
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {"path": {"type": "string"}},
            ...         "required": ["path"],
            ...     },
            ... )
            >>> converted = ChatCompletionsProvider._tools_to_chat([tool])
            >>> converted[0]["type"]
            'function'
            >>> converted[0]["function"]["parameters"] == tool.input_schema
            True
            >>> ChatCompletionsProvider._tools_to_chat(
            ...     [tool.model_copy(update={"enabled": False})]
            ... )
            []
        """
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
        """把 Chat Completions 原始 chunk 流转换成内部事件流。

        文本片段会立即转发；累计 Token 用量保留最后一个快照；工具调用按
        ``index`` 聚合身份字段和 JSON 参数片段。只有流包含合法
        ``finish_reason`` 时，才会产出最终的 ``ModelCompletedEvent``。

        Args:
            stream: OpenAI 兼容 SDK 返回的异步 chunk 迭代器。chunk 可以是 SDK
                对象，也可以是具有相同字段的字典。

        Yields:
            文本增量、可选 Token 用量、按 index 排序的工具调用，以及最后的
            成功终止事件。

        Raises:
            LLMProtocolError: choice、文本、工具调用、参数或结束原因不符合协议。
            LLMProviderError: 供应商使用 ``content_filter`` 拒绝响应。

        Example:
            >>> import asyncio
            >>> async def chunks():
            ...     yield {
            ...         "choices": [
            ...             {"index": 0, "delta": {"content": "你"}}
            ...         ]
            ...     }
            ...     yield {
            ...         "choices": [
            ...             {
            ...                 "index": 0,
            ...                 "delta": {"content": "好"},
            ...                 "finish_reason": "stop",
            ...             }
            ...         ]
            ...     }
            >>> async def event_types():
            ...     provider = ChatCompletionsProvider(
            ...         client=object(), api_key_env=None
            ...     )
            ...     return [
            ...         event.type
            ...         async for event in provider._events_from_chat_stream(chunks())
            ...     ]
            >>> asyncio.run(event_types())
            ['message_delta', 'message_delta', 'completed']
        """
        tool_parts: dict[int, dict[str, Any]] = {}
        latest_usage: dict[str, int] = {}
        finish_reason: str | None = None

        async for chunk in stream:
            usage = self._chat_usage(chunk)
            if usage:
                latest_usage = usage

            choice = self._stream_choice(chunk)
            if choice is None:
                continue

            finish_reason = self._stream_finish_reason(finish_reason, choice)
            content = self._stream_content(choice)
            if content:
                yield ModelMessageDeltaEvent(
                    payload={"text": content},
                )
            self._merge_stream_tool_calls(tool_parts, choice)

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
            yield ModelTokenCountEvent(payload=latest_usage)
        for call in calls:
            yield ModelToolCallEvent(payload=call)
        yield ModelCompletedEvent()

    @staticmethod
    def _stream_choice(chunk: Any) -> Any | None:
        """从一个流式 chunk 中取得唯一受支持的 choice。

        Args:
            chunk: SDK chunk 对象或具有相同字段的字典。

        Returns:
            index 为 ``0`` 的唯一 choice；usage-only 等无 choice chunk 返回
            ``None``。

        Raises:
            LLMProtocolError: chunk 含多个 choice，或唯一 choice 的 index 不是 0。

        Example:
            >>> choice = ChatCompletionsProvider._stream_choice(
            ...     {"choices": [{"index": 0, "delta": {"content": "你好"}}]}
            ... )
            >>> choice["delta"]["content"]
            '你好'
            >>> ChatCompletionsProvider._stream_choice({"choices": []}) is None
            True
        """
        choices = list(get_field(chunk, "choices", []) or [])
        if len(choices) > 1:
            raise LLMProtocolError("multiple chat choices are not supported")
        if not choices:
            return None

        choice = choices[0]
        if get_field(choice, "index", 0) != 0:
            raise LLMProtocolError("multiple chat choices are not supported")
        return choice

    @staticmethod
    def _stream_finish_reason(previous: str | None, choice: Any) -> str | None:
        """累积 choice 的结束原因，并拒绝流内互相矛盾的终态。

        Args:
            previous: 先前 chunk 已确认的结束原因，尚未出现时为 ``None``。
            choice: 当前 SDK choice 对象或字典。

        Returns:
            当前 choice 没有结束原因时返回 ``previous``；否则返回字符串形式的
            当前结束原因。

        Raises:
            LLMProtocolError: 当前结束原因与先前非空原因不同。

        Example:
            >>> ChatCompletionsProvider._stream_finish_reason(
            ...     None, {"finish_reason": None}
            ... ) is None
            True
            >>> ChatCompletionsProvider._stream_finish_reason(
            ...     None, {"finish_reason": "stop"}
            ... )
            'stop'
        """
        reason = get_field(choice, "finish_reason")
        if reason is None:
            return previous
        if previous is not None and previous != reason:
            raise LLMProtocolError("chat finish_reason changed within a stream")
        return str(reason)

    @staticmethod
    def _stream_content(choice: Any) -> str | None:
        """读取当前 choice 的可展示文本增量。

        Args:
            choice: 当前 SDK choice 对象或字典。

        Returns:
            非空文本片段；没有 content 或 content 为空字符串时返回 ``None``。

        Raises:
            LLMProtocolError: content 存在但不是字符串。

        Example:
            >>> ChatCompletionsProvider._stream_content(
            ...     {"delta": {"content": "你好"}}
            ... )
            '你好'
            >>> ChatCompletionsProvider._stream_content(
            ...     {"delta": {"content": ""}}
            ... ) is None
            True
        """
        delta = get_field(choice, "delta", {}) or {}
        content = get_field(delta, "content")
        if content is None:
            return None
        if not isinstance(content, str):
            raise LLMProtocolError("chat content delta must be text")
        return content or None

    @staticmethod
    def _merge_stream_tool_calls(
        tool_parts: dict[int, dict[str, Any]],
        choice: Any,
    ) -> None:
        """把当前 choice 的工具调用分片合并到跨 chunk 缓冲区。

        Args:
            tool_parts: 以非负调用 index 为键的可变聚合状态；本方法会原地更新。
            choice: 当前 SDK choice 对象或具有相同字段的字典。

        Returns:
            ``None``。工具身份和参数片段直接写入 ``tool_parts``。

        Raises:
            LLMProtocolError: index 非法，身份字段不是文本或发生变化，或者参数
                分片不是字符串。

        Example:
            >>> parts = {}
            >>> ChatCompletionsProvider._merge_stream_tool_calls(
            ...     parts,
            ...     {
            ...         "delta": {
            ...             "tool_calls": [
            ...                 {
            ...                     "index": 0,
            ...                     "id": "call_read",
            ...                     "function": {
            ...                         "name": "read_file",
            ...                         "arguments": '{"pa',
            ...                     },
            ...                 }
            ...             ]
            ...         }
            ...     },
            ... )
            >>> ChatCompletionsProvider._merge_stream_tool_calls(
            ...     parts,
            ...     {
            ...         "delta": {
            ...             "tool_calls": [
            ...                 {
            ...                     "index": 0,
            ...                     "function": {"arguments": 'th":"README.md"}'},
            ...                 }
            ...             ]
            ...         }
            ...     },
            ... )
            >>> "".join(parts[0]["arguments"])
            '{"path":"README.md"}'
        """
        delta = get_field(choice, "delta", {}) or {}
        for raw_call in get_field(delta, "tool_calls", []) or []:
            index = get_field(raw_call, "index", 0)
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
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
                    raise LLMProtocolError("streamed tool arguments must be strings")
                part["arguments"].append(arguments)

    def _events_from_chat_response(self, response: Any) -> list[ModelEvent]:
        """把一个完整 Chat Completions 响应转换成内部事件列表。

        Args:
            response: OpenAI 兼容 SDK 的完整响应对象，或具有相同字段的字典。

        Returns:
            按单个文本 part、可选 Token 用量、工具调用和成功终止标志排列的
            事件列表。空文本不会产生消息事件。

        Raises:
            LLMProtocolError: choice 数量、索引、文本、工具调用、usage 或结束原因
                不符合内部协议。
            LLMProviderError: 供应商使用 ``content_filter`` 拒绝响应。

        Example:
            >>> provider = ChatCompletionsProvider(
            ...     client=object(), api_key_env=None
            ... )
            >>> events = provider._events_from_chat_response(
            ...     {
            ...         "choices": [
            ...             {
            ...                 "index": 0,
            ...                 "message": {"content": "你好"},
            ...                 "finish_reason": "stop",
            ...             }
            ...         ],
            ...         "usage": {
            ...             "prompt_tokens": 2,
            ...             "completion_tokens": 1,
            ...             "total_tokens": 3,
            ...         },
            ...     }
            ... )
            >>> [event.type for event in events]
            ['message_delta', 'token_count', 'completed']
            >>> events[0].payload.text
            '你好'
        """
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
                    ModelMessageDeltaEvent(
                        payload={"text": content},
                    )
                )
        usage = self._chat_usage(response)
        if usage:
            events.append(ModelTokenCountEvent(payload=usage))
        events.extend(ModelToolCallEvent(payload=call) for call in calls)
        events.append(ModelCompletedEvent())
        return events

    @staticmethod
    def _validate_finish_reason(reason: Any, has_tool_calls: bool) -> None:
        """确认供应商结束原因表示一次完整且自洽的响应。

        Args:
            reason: Chat Completions choice 的最终 ``finish_reason``。
            has_tool_calls: 转换后的响应是否包含至少一个工具调用。

        Returns:
            ``None``。只有无工具调用的 ``stop`` 和有工具调用的
            ``tool_calls`` 会成功返回。

        Raises:
            LLMProtocolError: 结束原因缺失、达到 Token 上限、不受支持，或与工具
                调用存在性不一致。
            LLMProviderError: 响应被供应商内容过滤器阻止。

        Example:
            >>> ChatCompletionsProvider._validate_finish_reason("stop", False)
            >>> ChatCompletionsProvider._validate_finish_reason("tool_calls", True)
        """
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
        """把不同 Chat 兼容端点的 usage 字段归一成内部 Token 口径。

        Args:
            response: 含可选 ``usage`` 的完整响应或流式 chunk。

        Returns:
            没有 usage 时返回空字典；否则返回 input、output、reasoning、cached
            input 和 total 五个非负整数。

        Raises:
            LLMProtocolError: Token 字段不是非负整数，或供应商 total 不等于
                input 与 output 之和。

        Example:
            >>> ChatCompletionsProvider._chat_usage(
            ...     {
            ...         "usage": {
            ...             "prompt_tokens": 7,
            ...             "completion_tokens": 3,
            ...             "total_tokens": 10,
            ...             "prompt_tokens_details": {"cached_tokens": 2},
            ...             "completion_tokens_details": {"reasoning_tokens": 1},
            ...         }
            ...     }
            ... )
            {'input_tokens': 7, 'output_tokens': 3, 'reasoning_tokens': 1, 'cached_input_tokens': 2, 'total_tokens': 10}
        """
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
        if total_tokens != input_tokens + output_tokens:
            raise LLMProtocolError(
                "usage.total_tokens must equal input_tokens + output_tokens"
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
    """把一条内部消息转换成一个 Chat Completions message item。

    Args:
        message: 文本、单个工具调用或工具结果消息。

    Returns:
        与具体消息类型对应的 Chat Completions 字典。

    Raises:
        TypeError: 运行时传入了 ``ModelMessage`` 联合之外的对象。

    Example:
        >>> _message_to_chat_item(
        ...     ModelToolResultMessage(
        ...         tool_call_id="call_read", content="file contents"
        ...     )
        ... )
        {'role': 'tool', 'tool_call_id': 'call_read', 'content': 'file contents'}
    """
    if isinstance(message, ModelToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if isinstance(message, ModelToolCallMessage):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_message_to_chat_tool_call(message)],
        }
    if not isinstance(message, ModelTextMessage):
        raise TypeError(f"unsupported model message: {type(message).__name__}")
    return {"role": message.role.value, "content": message.content}


def _message_to_chat_tool_call(message: ModelToolCallMessage) -> dict[str, Any]:
    """把内部工具调用转换成 Chat Completions function call。

    Args:
        message: assistant 产生的结构化工具调用消息。

    Returns:
        含 call ID、function 类型、名称和紧凑 JSON 参数的字典。

    Raises:
        TypeError: arguments 含无法 JSON 序列化的值。

    Example:
        >>> _message_to_chat_tool_call(
        ...     ModelToolCallMessage(
        ...         tool_call_id="call_read",
        ...         name="read_file",
        ...         arguments={"path": "README.md"},
        ...     )
        ... )["function"]
        {'name': 'read_file', 'arguments': '{"path":"README.md"}'}
    """
    return {
        "id": message.tool_call_id,
        "type": "function",
        "function": {
            "name": message.name,
            "arguments": serialize_arguments(message.arguments),
        },
    }


def _chat_tool_calls(message: Any) -> list[ToolCall]:
    """解析完整 assistant message 中的全部 Chat 工具调用。

    Args:
        message: SDK assistant message 对象或具有相同字段的字典。

    Returns:
        保持供应商顺序的内部 ``ToolCall`` 列表。

    Raises:
        LLMProtocolError: call ID、名称或参数非法，或同一消息包含重复 call ID。

    Example:
        >>> calls = _chat_tool_calls(
        ...     {
        ...         "tool_calls": [
        ...             {
        ...                 "id": "call_read",
        ...                 "function": {
        ...                     "name": "read_file",
        ...                     "arguments": '{"path":"README.md"}',
        ...                 },
        ...             }
        ...         ]
        ...     }
        ... )
        >>> calls[0].model_dump(mode="json")
        {'call_id': 'call_read', 'name': 'read_file', 'arguments': {'path': 'README.md'}}
    """
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


def _set_tool_part(
    part: dict[str, Any],
    field: Literal["call_id", "name"],
    value: Any,
) -> None:
    """合并一个流式工具身份字段，并拒绝非空值发生变化。

    DashScope 可能在续传 chunk 中发送空字符串占位符；空值不会覆盖已经确认的
    call ID 或工具名。

    Args:
        part: 一个工具调用的可变聚合状态。
        field: 要更新的 ``call_id`` 或 ``name`` 字段。
        value: 当前 chunk 提供的候选值。

    Returns:
        ``None``。有效的非空值会原地写入 ``part``。

    Raises:
        LLMProtocolError: 非空值不是字符串，或与先前确认的值不同。

    Example:
        >>> part = {"call_id": None, "name": None, "arguments": []}
        >>> _set_tool_part(part, "call_id", "call_read")
        >>> _set_tool_part(part, "call_id", "")
        >>> part["call_id"]
        'call_read'
    """
    if value == "":
        return
    if not isinstance(value, str):
        raise LLMProtocolError(f"streamed tool call {field} must be text")
    previous = part[field]
    if previous is not None and previous != value:
        raise LLMProtocolError(f"streamed tool call changed its {field}")
    part[field] = value
