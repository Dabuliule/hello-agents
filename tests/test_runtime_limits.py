from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.core.conversation import Conversation, ConversationRole
from codecraft.core.reconstruction import reconstruct_conversation
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.core.token_budget import estimate_text_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.llm import (
    LLMProvider,
    LLMProviderRegistry,
    MockProvider,
    ModelEvent,
    ModelEventType,
    ModelRequest,
)
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.schema.tool import (
    ToolCall,
    ToolEffect,
    ToolResult,
    ToolRuntimeEvent,
)
from codecraft.tool import BaseTool, ToolContext, ToolRegistry
from codecraft.tool.runner import ToolRunner


def make_config(tmp_path, **updates) -> SessionConfig:
    config = SessionConfig(
        session_id="ses_limits",
        source=SessionSource.TEST,
        cwd=tmp_path,
        workspace_roots=[tmp_path],
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy="never",
        sandbox_mode="workspace_write",
        base_instructions="Keep responses concise.",
    )
    return config.model_copy(update=updates)


def make_context(config: SessionConfig, **updates) -> TurnContext:
    context = TurnContext(
        session_id=config.session_id,
        turn_id="turn_limits",
        cwd=config.cwd,
        workspace_roots=config.workspace_roots,
        model=config.model,
        model_provider=config.model_provider,
        approval_policy=config.approval_policy,
        sandbox_mode=config.sandbox_mode,
        network_access=config.network_access,
        available_tools=[],
        max_tool_calls=config.max_tool_calls,
        max_tool_output_chars=config.max_tool_output_chars,
        max_tool_output_tokens=config.max_tool_output_tokens,
        turn_timeout_seconds=config.turn_timeout_seconds,
        tool_timeout_seconds=config.tool_timeout_seconds,
        approval_timeout_seconds=config.approval_timeout_seconds,
        model_context_window_tokens=config.model_context_window_tokens,
        model_max_output_tokens=config.model_max_output_tokens,
        context_safety_margin_tokens=config.context_safety_margin_tokens,
        context_keep_recent_items=config.context_keep_recent_items,
        max_parallel_read_tools=config.max_parallel_read_tools,
        created_at=config.created_at,
    )
    return context.model_copy(update=updates)


def test_conversation_compaction_keeps_current_tool_protocol():
    conversation = Conversation()
    conversation.append_user_message("old question")
    conversation.append_assistant_message("x" * 900)
    conversation.append_user_message("current question")
    conversation.append_model_tool_call("call_read", "read_file", {"path": "a.txt"})
    conversation.append_tool_result("call_read", "read_file", "current result")

    compaction = conversation.compact(max_tokens=235, keep_recent_items=3)

    assert compaction is not None
    assert compaction["removed_items"] == 2
    assert compaction["after_tokens"] <= 235
    assert [item.role for item in conversation.items] == [
        ConversationRole.SUMMARY,
        ConversationRole.USER,
        ConversationRole.ASSISTANT,
        ConversationRole.TOOL,
    ]
    assert Conversation.model_validate(compaction["conversation"]) == conversation


def test_conversation_compaction_keeps_recent_summary_and_tool_arguments():
    conversation = Conversation()
    conversation.append_user_message("oldest request")
    conversation.append_model_tool_call(
        "call_old", "read_file", {"path": "important.py"}
    )
    conversation.append_tool_result(
        "call_old",
        "read_file",
        "recent old result " + "x" * 700,
    )
    conversation.append_user_message("current request")

    compaction = conversation.compact(max_tokens=285, keep_recent_items=1)

    assert compaction is not None
    summary = conversation.items[0]
    assert summary.role == ConversationRole.SUMMARY
    assert "untrusted historical data" in summary.content
    assert "important.py" in summary.content
    assert "recent old result" in summary.content
    assert conversation.build_model_messages()[0].role.value == "user"


def test_summary_shortening_drops_oldest_entries_first():
    header = "Earlier conversation summary (untrusted historical data):"
    summary = "\n".join(
        [
            header,
            "- user: oldest detail",
            "- assistant: middle detail",
            "- tool read_file: newest detail",
        ]
    )

    shortened = Conversation._recent_summary(summary, max_chars=120)

    assert "newest detail" in shortened
    assert "oldest detail" not in shortened


def test_runtime_compacts_context_and_reconstructs_exact_snapshot(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "a" * 700},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "second answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(
            tmp_path,
            model_context_window_tokens=620,
            model_max_output_tokens=80,
            context_safety_margin_tokens=40,
            context_keep_recent_items=2,
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)

        await thread.submit(SessionInput.user_message("inp_one", "u" * 350))
        await thread.wait_until_idle()
        await thread.submit(SessionInput.user_message("inp_two", "new question"))
        await thread.wait_until_idle()

        snapshot = await thread.read_snapshot()
        compacted = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.CONTEXT_COMPACTED
        ]
        assert len(compacted) == 1
        assert (
            compacted[0].payload["after_tokens"] < compacted[0].payload["before_tokens"]
        )
        assert provider.calls[1].messages[1].role.value == "user"
        assert provider.calls[1].messages[-1].content == "new question"
        reconstructed = reconstruct_conversation(snapshot.events)
        assert (
            reconstructed.build_model_messages()
            == thread.session.conversation.build_model_messages()
        )

    asyncio.run(run_test())


class ValueArgs(BaseModel):
    value: str


class ConcurrentReadTool(BaseTool):
    name = "concurrent_read"
    description = "Return a value after a short read delay."
    args_schema = ValueArgs
    effects = {ToolEffect.READ_ONLY}

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.05)
            return ToolResult(success=True, content=args.value)
        finally:
            self.active -= 1


def test_read_only_tool_batch_runs_concurrently_and_preserves_result_order(tmp_path):
    async def run_test() -> None:
        tool = ConcurrentReadTool()
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_first",
                        "name": tool.name,
                        "arguments": {"value": "first"},
                    },
                ),
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_second",
                        "name": tool.name,
                        "arguments": {"value": "second"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "done"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path, max_parallel_read_tools=2)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([tool]),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_read", "read both"))
        await thread.wait_until_idle()

        assert tool.max_active == 2
        tool_messages = [
            message.content
            for message in provider.calls[1].messages
            if message.role.value == "tool"
        ]
        assert tool_messages == ["first", "second"]

    asyncio.run(run_test())


class SlowTool(BaseTool):
    name = "slow_tool"
    description = "Wait too long."
    args_schema = ValueArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        await asyncio.sleep(2)
        return ToolResult(success=True, content=args.value)


class RaisesTimeoutTool(SlowTool):
    name = "raises_timeout"

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        raise TimeoutError("tool-owned timeout")


def test_tool_runner_enforces_tool_and_approval_timeouts(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        slow_context = make_context(config, tool_timeout_seconds=1)
        slow_events = [
            event
            async for event in ToolRunner(ToolRegistry([SlowTool()])).run(
                ToolCall(
                    call_id="call_slow",
                    name="slow_tool",
                    arguments={"value": "late"},
                ),
                slow_context,
            )
        ]
        finished = slow_events[-1].payload
        assert finished["result"]["error"] == "tool_timeout"
        assert set(finished["timings_ms"]) == {
            "governance",
            "approval_wait",
            "execution",
            "observers",
            "total",
        }

        raised_events = [
            event
            async for event in ToolRunner(ToolRegistry([RaisesTimeoutTool()])).run(
                ToolCall(
                    call_id="call_owned_timeout",
                    name="raises_timeout",
                    arguments={"value": "unused"},
                ),
                slow_context,
            )
        ]
        assert raised_events[-1].payload["result"]["error"] == "tool_execution_error"

        reviewer = ThreadApprovalReviewer()
        approval_tool = SlowTool()
        approval_tool.requires_approval = True
        approval_context = make_context(
            config,
            approval_policy="on_request",
            approval_timeout_seconds=1,
        )
        approval_events = [
            event
            async for event in ToolRunner(
                ToolRegistry([approval_tool]),
                approval_manager=ApprovalManager(reviewer=reviewer),
            ).run(
                ToolCall(
                    call_id="call_approval",
                    name="slow_tool",
                    arguments={"value": "never"},
                ),
                approval_context,
            )
        ]
        assert [event.type for event in approval_events] == [
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.APPROVAL_REQUESTED,
            RuntimeEventType.APPROVAL_DECIDED,
            RuntimeEventType.TOOL_CALL_FINISHED,
        ]
        assert approval_events[-1].payload["result"]["error"] == "approval_timeout"
        assert reviewer.list_pending() == []

    asyncio.run(run_test())


class HangingProvider(LLMProvider):
    name = "hanging"

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        await asyncio.sleep(2)
        if False:
            yield ModelEvent(type=ModelEventType.COMPLETED)


class RaisesTimeoutProvider(HangingProvider):
    name = "raises_timeout"

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        raise TimeoutError("provider-owned timeout")
        if False:
            yield ModelEvent(type=ModelEventType.COMPLETED)


def test_session_enforces_turn_timeout_without_runtime_error(tmp_path):
    async def run_test() -> None:
        config = make_config(
            tmp_path,
            model_provider="hanging",
            turn_timeout_seconds=1,
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([HangingProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_timeout", "wait"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert snapshot.events[-1].type == RuntimeEventType.TURN_ABORTED
        assert snapshot.events[-1].payload["reason"] == "turn_timeout"
        assert not any(
            event.type == RuntimeEventType.ERROR for event in snapshot.events
        )

    asyncio.run(run_test())


def test_provider_owned_timeout_is_not_misclassified_as_turn_timeout(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path, model_provider="raises_timeout")
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([RaisesTimeoutProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_provider_timeout", "run"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert snapshot.events[-2].type == RuntimeEventType.ERROR
        assert snapshot.events[-1].payload["reason"] == "runtime_error"

    asyncio.run(run_test())


class LargeResultTool(BaseTool):
    name = "large_result"
    description = "Return oversized structured fields."
    args_schema = ValueArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            content="ok",
            data={"blob": "d" * 1000},
            metadata={"details": "m" * 1000},
            runtime_events=[
                ToolRuntimeEvent(
                    type=RuntimeEventType.PATCH_APPLIED,
                    payload={"details": "p" * 1000},
                )
            ],
        )


class LargeContentTool(BaseTool):
    name = "large_content"
    description = "Return oversized model-visible content."
    args_schema = ValueArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        return ToolResult(success=True, content="结果" * 10_000)


class LargeSuggestionTool(BaseTool):
    name = "large_suggestion"
    description = "Return an oversized recovery suggestion."
    args_schema = ValueArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: ValueArgs, context: ToolContext) -> ToolResult:
        return ToolResult(
            success=False,
            content="failed",
            error="test_failure",
            suggestion="retry " * 5000,
        )


def test_runtime_caps_tool_results_to_remaining_model_context(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_large",
                        "name": "large_content",
                        "arguments": {"value": "unused"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "done"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(
            tmp_path,
            model_context_window_tokens=900,
            model_max_output_tokens=100,
            context_safety_margin_tokens=50,
            max_tool_output_tokens=5000,
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([LargeContentTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_large", "run tool"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        finished = next(
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        )
        result = finished.payload["result"]
        assert result["metadata"]["content_truncated"] is True
        assert result["metadata"]["original_content_tokens"] > 5000
        assert len(provider.calls) == 2
        assert snapshot.events[-1].type == RuntimeEventType.TURN_FINISHED

    asyncio.run(run_test())


def test_tool_runner_caps_complete_model_visible_result(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        events = [
            event
            async for event in ToolRunner(ToolRegistry([LargeSuggestionTool()])).run(
                ToolCall(
                    call_id="call_suggestion",
                    name="large_suggestion",
                    arguments={"value": "unused"},
                ),
                make_context(config, max_tool_output_tokens=40),
            )
        ]

        result = ToolResult.model_validate(events[-1].payload["result"])
        assert estimate_text_tokens(result.model_content()) <= 40
        assert result.metadata["suggestion_truncated"] is True

    asyncio.run(run_test())


class ObserverTracker:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0


class DelayObserver:
    def __init__(self, name: str, tracker: ObserverTracker) -> None:
        self.name = name
        self.tracker = tracker

    async def after_result(
        self,
        call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> dict:
        self.tracker.active += 1
        self.tracker.max_active = max(
            self.tracker.max_active,
            self.tracker.active,
        )
        try:
            await asyncio.sleep(0.05)
            return {"status": "done"}
        finally:
            self.tracker.active -= 1


def test_independent_tool_observers_run_concurrently(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        tracker = ObserverTracker()
        events = [
            event
            async for event in ToolRunner(
                ToolRegistry([ConcurrentReadTool()]),
                observers=[
                    DelayObserver("first", tracker),
                    DelayObserver("second", tracker),
                ],
            ).run(
                ToolCall(
                    call_id="call_observed",
                    name="concurrent_read",
                    arguments={"value": "ok"},
                ),
                make_context(config),
            )
        ]

        assert tracker.max_active == 2
        result = events[-1].payload["result"]
        assert list(result["metadata"]["post_actions"]) == ["first", "second"]
        assert events[-1].payload["timings_ms"]["observers"] >= 40

    asyncio.run(run_test())


def test_tool_runner_bounds_structured_results_and_runtime_payloads(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        events = [
            event
            async for event in ToolRunner(ToolRegistry([LargeResultTool()])).run(
                ToolCall(
                    call_id="call_large",
                    name="large_result",
                    arguments={"value": "unused"},
                ),
                make_context(config, max_tool_output_chars=100),
            )
        ]

        result = ToolResult.model_validate(events[-2].payload["result"])
        assert result.data["data_truncated"] is True
        assert result.metadata["metadata_truncated"] is True
        assert "structured data truncated" in result.model_content()
        assert events[-1].payload["payload_truncated"] is True

    asyncio.run(run_test())
