from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from codecraft.core.turn import TurnStatus
from codecraft.core.event_bus import EventBus
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.llm import (
    LLMProvider,
    LLMProviderError,
    LLMProviderRegistry,
    MockProvider,
    ModelCompletedEvent,
    ModelEvent,
    ModelMessageCompletedEvent,
    ModelRequest,
    ModelToolCallEvent,
    QwenProvider,
)
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ReadFileTool, ToolRegistry


def make_config(tmp_path) -> SessionConfig:
    return SessionConfig(
        session_id="ses_lifecycle",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy="never",
        sandbox_mode="workspace_write",
    )


class BlockingThenCompleteProvider(LLMProvider):
    name = "blocking"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.call_count = 0

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        self.call_count += 1
        if self.call_count == 1:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        else:
            yield ModelMessageCompletedEvent(
                payload={"text": "continued after interrupt"},
            )
            yield ModelCompletedEvent()


def test_interrupt_cancels_active_provider_before_starting_next_turn(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"model_provider": "blocking"})
        provider = BlockingThenCompleteProvider()
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "block"))
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        await asyncio.wait_for(thread.interrupt("test_interrupt"), timeout=1)

        assert provider.cancelled.is_set()
        assert thread.session.active_turn is None
        assert thread.session.status.value == "idle"
        snapshot = await thread.read_snapshot()
        aborted = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TURN_ABORTED
        ]
        assert len(aborted) == 1
        assert aborted[0].payload["reason"] == "test_interrupt"

        await thread.submit(SessionInput.user_message("inp_two", "continue"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()
        assert snapshot.events[-1].type == RuntimeEventType.TURN_FINISHED
        assert snapshot.events[-1].payload["answer"] == "continued after interrupt"

    asyncio.run(run_test())


def test_close_waits_for_active_turn_before_session_closed_event(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"model_provider": "blocking"})
        provider = BlockingThenCompleteProvider()
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "block"))
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        await asyncio.wait_for(thread.close(), timeout=1)

        assert provider.cancelled.is_set()
        assert thread.session.active_turn is None
        snapshot = await thread.read_snapshot()
        assert snapshot.events[-2].type == RuntimeEventType.TURN_ABORTED
        assert snapshot.events[-2].payload["reason"] == "session_closed"
        assert snapshot.events[-1].type == RuntimeEventType.SESSION_CLOSED

    asyncio.run(run_test())


def test_close_does_not_cancel_interrupt_cleanup(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"model_provider": "blocking"})
        provider = BlockingThenCompleteProvider()
        abort_started = asyncio.Event()
        release_abort = asyncio.Event()
        event_bus = EventBus()

        async def delay_abort(event) -> None:
            if event.type == RuntimeEventType.TURN_ABORTED:
                abort_started.set()
                await release_abort.wait()

        event_bus.subscribe(delay_abort)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
            event_bus=event_bus,
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "block"))
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        interrupt_task = asyncio.create_task(thread.interrupt("test_interrupt"))
        await asyncio.wait_for(abort_started.wait(), timeout=1)
        close_task = asyncio.create_task(thread.close())
        await asyncio.sleep(0)

        assert not interrupt_task.done()
        assert not close_task.done()

        release_abort.set()
        await asyncio.wait_for(
            asyncio.gather(interrupt_task, close_task),
            timeout=1,
        )

        snapshot = await thread.read_snapshot()
        terminal_events = [
            event.type
            for event in snapshot.events
            if event.type
            in {RuntimeEventType.TURN_ABORTED, RuntimeEventType.SESSION_CLOSED}
        ]
        assert terminal_events == [
            RuntimeEventType.TURN_ABORTED,
            RuntimeEventType.SESSION_CLOSED,
        ]

    asyncio.run(run_test())


def test_runtime_executes_all_tool_calls_from_one_model_response(tmp_path):
    async def run_test() -> None:
        (tmp_path / "one.txt").write_text("one", encoding="utf-8")
        (tmp_path / "two.txt").write_text("two", encoding="utf-8")
        provider = MockProvider(
            [
                ModelToolCallEvent(
                    payload={
                        "call_id": "call_one",
                        "name": "read_file",
                        "arguments": {"path": "one.txt"},
                    },
                ),
                ModelToolCallEvent(
                    payload={
                        "call_id": "call_two",
                        "name": "read_file",
                        "arguments": {"path": "two.txt"},
                    },
                ),
                ModelCompletedEvent(),
                ModelMessageCompletedEvent(
                    payload={"text": "read both files"},
                ),
                ModelCompletedEvent(),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "read both"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        model_calls = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.MODEL_TOOL_CALL
        ]
        tool_results = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        ]
        assert [event.payload["call_id"] for event in model_calls] == [
            "call_one",
            "call_two",
        ]
        assert [event.payload["result"]["content"] for event in tool_results] == [
            "one",
            "two",
        ]
        assert snapshot.events[-1].payload["tool_calls"] == 2
        assert len(provider.calls) == 2

        event_types = [event.type for event in snapshot.events]
        first_call = event_types.index(RuntimeEventType.MODEL_TOOL_CALL)
        first_start = event_types.index(RuntimeEventType.TOOL_CALL_STARTED)
        assert event_types[first_call:first_start] == [
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.MODEL_TOOL_CALL,
        ]

        chat_messages = QwenProvider._messages_to_chat(provider.calls[1].messages)
        assistant_batch = next(
            message for message in chat_messages if message.get("tool_calls")
        )
        assert [item["id"] for item in assistant_batch["tool_calls"]] == [
            "call_one",
            "call_two",
        ]
        assert [
            message["tool_call_id"]
            for message in chat_messages
            if message["role"] == "tool"
        ] == ["call_one", "call_two"]

    asyncio.run(run_test())


def test_runtime_rejects_over_budget_tool_batch_without_partial_execution(tmp_path):
    async def run_test() -> None:
        (tmp_path / "one.txt").write_text("one", encoding="utf-8")
        provider = MockProvider(
            [
                ModelToolCallEvent(
                    payload={
                        "call_id": "call_one",
                        "name": "read_file",
                        "arguments": {"path": "one.txt"},
                    },
                ),
                ModelToolCallEvent(
                    payload={
                        "call_id": "call_two",
                        "name": "read_file",
                        "arguments": {"path": "one.txt"},
                    },
                ),
                ModelCompletedEvent(),
            ]
        )
        config = make_config(tmp_path).model_copy(update={"max_tool_calls": 1})
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "read twice"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert not any(
            event.type == RuntimeEventType.MODEL_TOOL_CALL for event in snapshot.events
        )
        assert snapshot.events[-1].type == RuntimeEventType.TURN_ABORTED
        assert snapshot.events[-1].payload["reason"] == "max_tool_calls_exceeded"
        assert snapshot.events[-1].payload["tool_calls"] == 0
        assert [
            call["call_id"]
            for call in snapshot.events[-1].payload["metadata"]["requested_tool_calls"]
        ] == ["call_one", "call_two"]

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "script",
    [
        [
            ModelMessageCompletedEvent(
                payload={"text": "unterminated"},
            )
        ],
        [ModelCompletedEvent()],
    ],
)
def test_runtime_rejects_incomplete_or_empty_model_responses(tmp_path, script):
    class IncompleteProvider(LLMProvider):
        name = "mock"

        async def stream(self, request):
            for event in script:
                yield event

    async def run_test() -> None:
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([IncompleteProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "answer me"))
        turn = thread.session.active_turn
        assert turn is not None
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert turn.status == TurnStatus.ABORTED
        assert snapshot.events[-1].type == RuntimeEventType.TURN_ABORTED
        assert snapshot.events[-1].payload["reason"] == "model_protocol_error"
        assert snapshot.events[-1].payload["duration_ms"] >= 0

    asyncio.run(run_test())


def test_runtime_classifies_provider_exceptions_as_model_errors(tmp_path):
    class FailingProvider(LLMProvider):
        name = "failing"

        async def stream(self, request):
            if False:
                yield
            raise LLMProviderError("provider unavailable")

    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"model_provider": "failing"})
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([FailingProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "answer me"))
        turn = thread.session.active_turn
        assert turn is not None
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert turn.status == TurnStatus.ABORTED
        assert snapshot.events[-2].type == RuntimeEventType.ERROR
        assert snapshot.events[-2].payload["code"] == "model_error"
        assert snapshot.events[-1].payload["reason"] == "model_error"

    asyncio.run(run_test())
