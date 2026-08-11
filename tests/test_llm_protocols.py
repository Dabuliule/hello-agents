from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.llm import (
    LLMConfigError,
    LLMProtocolError,
    LLMProvider,
    LLMProviderError,
    LLMProviderRegistry,
    MockProvider,
    ModelCompletedEvent,
    ModelEvent,
    ModelMessageCompletedEvent,
    ModelMessageDeltaEvent,
    ModelRequest,
    ModelRole,
    ModelTextMessage,
    ModelTokenCountEvent,
    ModelToolCallEvent,
    ModelToolCallMessage,
    ModelToolResultMessage,
    OpenAIProvider,
    QwenProvider,
)
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ToolRegistry


class AsyncEvents:
    def __init__(
        self,
        events: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        self._events = iter(events)
        self._error = error

    def __aiter__(self) -> AsyncEvents:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._events)
        except StopIteration:
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
            raise StopAsyncIteration


class FakeResponses:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def create(self, **kwargs: Any) -> Any:
        return self.response


class FakeResponsesClient:
    def __init__(self, response: Any) -> None:
        self.responses = FakeResponses(response)


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response

    async def create(self, **kwargs: Any) -> Any:
        return self.response


class FakeChatClient:
    def __init__(self, response: Any) -> None:
        self.chat = type(
            "FakeChat",
            (),
            {"completions": FakeCompletions(response)},
        )()


def request() -> ModelRequest:
    return ModelRequest(
        model="test-model",
        messages=(ModelTextMessage(role=ModelRole.USER, content="hello"),),
    )


async def collect(provider: LLMProvider) -> list[ModelEvent]:
    return [event async for event in provider.stream(request())]


@pytest.mark.parametrize(
    "data",
    [
        {
            "type": "message",
            "role": ModelRole.TOOL,
            "content": "not a result",
        },
        {
            "type": "message",
            "role": ModelRole.USER,
            "content": "hello",
            "tool_call_id": "call_extra",
        },
        {
            "type": "tool_call",
            "role": ModelRole.USER,
            "name": "read_file",
            "tool_call_id": "call_read",
            "arguments": {},
        },
        {
            "type": "tool_call",
            "role": ModelRole.ASSISTANT,
            "content": "{}",
            "name": "read_file",
            "tool_call_id": "call_read",
            "arguments": {},
        },
        {
            "type": "tool_result",
            "role": ModelRole.TOOL,
            "content": "done",
        },
    ],
)
def test_model_message_rejects_illegal_field_combinations(data):
    with pytest.raises(ValidationError):
        ModelRequest.model_validate({"model": "test-model", "messages": [data]})


def test_model_message_discriminator_selects_concrete_variant():
    parsed = ModelRequest.model_validate(
        {
            "model": "test-model",
            "messages": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "tool_call",
                    "role": "assistant",
                    "name": "read_file",
                    "tool_call_id": "call_read",
                    "arguments": {"path": "README.md"},
                },
                {
                    "type": "tool_result",
                    "role": "tool",
                    "content": "done",
                    "tool_call_id": "call_read",
                },
            ],
        }
    )
    text, call, result = parsed.messages

    assert isinstance(text, ModelTextMessage)
    assert isinstance(call, ModelToolCallMessage)
    assert isinstance(result, ModelToolResultMessage)


def test_model_event_is_immutable():
    event = ModelMessageCompletedEvent(
        payload={"text": "done"},
    )

    with pytest.raises(ValidationError):
        event.payload.text = "changed"


def test_model_event_discriminator_selects_concrete_variant():
    adapter = TypeAdapter(ModelEvent)
    events = [
        adapter.validate_python(
            {"type": "message_delta", "payload": {"text": "hello"}}
        ),
        adapter.validate_python(
            {"type": "message_completed", "payload": {"text": "done"}}
        ),
        adapter.validate_python(
            {
                "type": "tool_call",
                "payload": {
                    "call_id": "call_read",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            }
        ),
        adapter.validate_python(
            {
                "type": "token_count",
                "payload": {"input_tokens": 7, "output_tokens": 3},
            }
        ),
        adapter.validate_python({"type": "completed"}),
    ]

    assert [type(event) for event in events] == [
        ModelMessageDeltaEvent,
        ModelMessageCompletedEvent,
        ModelToolCallEvent,
        ModelTokenCountEvent,
        ModelCompletedEvent,
    ]

    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "completed", "payload": {}})


def test_token_count_does_not_add_reasoning_twice():
    event = ModelTokenCountEvent(
        payload={
            "input_tokens": 7,
            "output_tokens": 3,
            "reasoning_tokens": 2,
        },
    )
    assert event.payload.total_tokens == 10

    with pytest.raises(ValidationError, match=r"input_tokens \+ output_tokens"):
        ModelTokenCountEvent(
            payload={
                "input_tokens": 7,
                "output_tokens": 3,
                "reasoning_tokens": 2,
                "total_tokens": 12,
            },
        )


def test_chat_usage_rejects_inconsistent_total_tokens():
    with pytest.raises(LLMProtocolError, match=r"input_tokens \+ output_tokens"):
        QwenProvider._chat_usage(
            {
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                }
            }
        )


def test_responses_usage_rejects_inconsistent_total_tokens():
    with pytest.raises(LLMProtocolError, match=r"input_tokens \+ output_tokens"):
        OpenAIProvider._usage(
            {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 11,
                }
            }
        )


def test_responses_stream_rejects_missing_terminal_event():
    provider = OpenAIProvider(
        client=FakeResponsesClient(
            AsyncEvents([{"type": "response.output_text.delta", "delta": "partial"}])
        )
    )

    with pytest.raises(LLMProtocolError, match="response.completed"):
        asyncio.run(collect(provider))


def test_responses_stream_raises_nested_provider_failure():
    provider = OpenAIProvider(
        client=FakeResponsesClient(
            AsyncEvents(
                [
                    {
                        "type": "response.failed",
                        "response": {
                            "status": "failed",
                            "error": {"message": "upstream unavailable"},
                        },
                    }
                ]
            )
        )
    )

    with pytest.raises(LLMProviderError, match="upstream unavailable"):
        asyncio.run(collect(provider))


def test_responses_rejects_incomplete_response():
    provider = OpenAIProvider(
        client=FakeResponsesClient(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            }
        )
    )

    with pytest.raises(LLMProtocolError, match="max_output_tokens"):
        asyncio.run(collect(provider))


def test_responses_rejects_invalid_tool_arguments():
    provider = OpenAIProvider(
        client=FakeResponsesClient(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": "{broken",
                    }
                ],
            }
        )
    )

    with pytest.raises(LLMProtocolError, match="invalid JSON"):
        asyncio.run(collect(provider))


@pytest.mark.parametrize(
    ("finish_reason", "message"),
    [
        (None, "without finish_reason"),
        ("length", "token limit"),
    ],
)
def test_chat_stream_requires_successful_finish_reason(finish_reason, message):
    choice: dict[str, Any] = {"index": 0, "delta": {"content": "partial"}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    provider = QwenProvider(client=FakeChatClient(AsyncEvents([{"choices": [choice]}])))

    with pytest.raises(LLMProtocolError, match=message):
        asyncio.run(collect(provider))


@pytest.mark.parametrize(
    ("finish_reason", "has_tool_calls"),
    [
        ("stop", False),
        ("tool_calls", True),
    ],
)
def test_chat_finish_reason_accepts_consistent_terminal_state(
    finish_reason,
    has_tool_calls,
):
    QwenProvider._validate_finish_reason(finish_reason, has_tool_calls)


@pytest.mark.parametrize(
    ("finish_reason", "has_tool_calls", "error_type", "message"),
    [
        ("content_filter", False, LLMProviderError, "content filter"),
        ("unknown", False, LLMProtocolError, "unsupported"),
        ("stop", True, LLMProtocolError, "does not match"),
        ("tool_calls", False, LLMProtocolError, "does not match"),
    ],
)
def test_chat_finish_reason_rejects_failure_or_mismatch(
    finish_reason,
    has_tool_calls,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        QwenProvider._validate_finish_reason(finish_reason, has_tool_calls)


def test_chat_stream_ignores_empty_role_chunk():
    provider = QwenProvider(
        client=FakeChatClient(
            AsyncEvents(
                [
                    {"choices": [{"index": 0, "delta": {"content": ""}}]},
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "done"},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                ]
            )
        )
    )

    events = asyncio.run(collect(provider))
    assert [event.type for event in events] == [
        "message_delta",
        "completed",
    ]


def test_chat_stream_ignores_empty_tool_identity_placeholders():
    provider = QwenProvider(
        client=FakeChatClient(
            AsyncEvents(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_list",
                                            "function": {
                                                "name": "list_directory",
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "",
                                            "function": {
                                                "name": "",
                                                "arguments": '{"path":"."}',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]
            )
        )
    )

    events = asyncio.run(collect(provider))

    assert [event.type for event in events] == [
        "tool_call",
        "completed",
    ]
    assert events[0].payload.model_dump(mode="json") == {
        "call_id": "call_list",
        "name": "list_directory",
        "arguments": {"path": "."},
    }


def test_chat_stream_rejects_conflicting_non_empty_tool_identity():
    provider = QwenProvider(
        client=FakeChatClient(
            AsyncEvents(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_first",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": "{}",
                                            },
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_second",
                                            "function": {},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                ]
            )
        )
    )

    with pytest.raises(LLMProtocolError, match="changed its call_id"):
        asyncio.run(collect(provider))


def test_provider_wraps_errors_raised_during_stream_iteration():
    provider = QwenProvider(
        client=FakeChatClient(AsyncEvents([], OSError("connection reset")))
    )

    with pytest.raises(LLMProviderError, match="connection reset"):
        asyncio.run(collect(provider))


def test_provider_reuses_and_closes_only_owned_client(monkeypatch):
    class FakeClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    created: list[FakeClient] = []

    def create_client(**kwargs: Any) -> FakeClient:
        client = FakeClient()
        created.append(client)
        return client

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", create_client)
    owned = QwenProvider(api_key="secret")
    assert owned._client() is owned._client()
    asyncio.run(owned.close())
    assert len(created) == 1
    assert created[0].close_calls == 1

    injected = FakeClient()
    asyncio.run(QwenProvider(client=injected).close())
    assert injected.close_calls == 0


def test_openai_provider_requires_its_configured_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()

    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        provider._client()
    with pytest.raises(LLMConfigError) as error:
        asyncio.run(collect(provider))
    assert error.value.code == "model_config_error"


def test_registry_normalizes_names_and_closes_every_provider():
    closed: list[str] = []

    class CloseProvider(LLMProvider):
        def __init__(self, name: str, fails: bool = False) -> None:
            self.name = name
            self.fails = fails

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelEvent]:
            if False:
                yield ModelCompletedEvent()

        async def close(self) -> None:
            closed.append(self.name)
            if self.fails:
                raise RuntimeError("close failed")

    first = CloseProvider("FIRST", fails=True)
    second = CloseProvider("second")
    registry = LLMProviderRegistry([first, second])

    assert registry.get(" first ") is first
    with pytest.raises(LLMProviderError, match="1 model provider"):
        asyncio.run(registry.close())
    assert closed == ["second", "FIRST"]
    with pytest.raises(LLMConfigError, match="missing"):
        registry.get("missing")


def test_unknown_provider_is_rejected_before_session_file_creation(tmp_path):
    config = SessionConfig(
        session_id="ses_unknown_provider",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="test-model",
        model_provider=" missing ",
        approval_policy="never",
        sandbox_mode="workspace_write",
    )
    runtime = AgentRuntime(
        session_store=SessionStore(config.codecraft_home),
        llm_providers=LLMProviderRegistry(),
        tool_registry=ToolRegistry(),
    )

    with pytest.raises(LLMConfigError):
        asyncio.run(runtime.create_thread(config))
    assert not (config.codecraft_home / "sessions").exists()


def test_mock_provider_requires_explicit_boundaries_and_snapshots_calls():
    with pytest.raises(ValueError, match="without completed"):
        MockProvider(
            [
                ModelMessageCompletedEvent(
                    payload={"text": "unfinished"},
                )
            ]
        )

    provider = MockProvider(
        [
            ModelMessageCompletedEvent(
                payload={"text": "done"},
            ),
            ModelCompletedEvent(),
        ]
    )
    model_request = request()

    async def run() -> None:
        assert [event async for event in provider.stream(model_request)]

    asyncio.run(run())
    assert provider.calls[0] == model_request
    assert provider.calls[0] is not model_request
