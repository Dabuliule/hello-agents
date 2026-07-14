from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
import json
from time import monotonic
from typing import TYPE_CHECKING, Any

from codecraft.core.conversation import Conversation
from codecraft.core.errors import CodecraftError
from codecraft.core.turn_context import TurnContext
from codecraft.llm.base import LLMProtocolError
from codecraft.llm.base import ModelRequest
from codecraft.llm.events import (
    ModelEventType,
    ModelTextPayload,
    ModelTokenCountPayload,
)
from codecraft.llm.messages import ModelMessage
from codecraft.prompt import PromptBuilder
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput, UserMessagePayload
from codecraft.schema.tool import ToolCall, ToolEffect, ToolResult

if TYPE_CHECKING:
    from codecraft.core.session import Session


class TurnStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class Turn:
    """一次用户输入对应的模型执行轮次。

    `Turn` 管理从用户消息进入对话、调用模型、执行 tool call，到产出最终
    assistant 消息的完整循环。它不负责持久化细节，所有外部可见状态都通过
    session event 发出去。
    """

    def __init__(
        self,
        *,
        session: Session,
        turn_id: str,
    ) -> None:
        self.turn_id = turn_id
        self.session = session
        self.context = self._build_context()
        self.status = TurnStatus.CREATED
        self.tool_call_count = 0
        self.prompt_builder = PromptBuilder()
        self._started_at: float | None = None

    async def run(self, user_input: SessionInput) -> None:
        """运行一次用户输入，直到模型给出最终回复或轮次中止。

        模型可能在一次响应中要求调用工具；工具结果会回填到 conversation，
        然后继续下一次模型调用，直到没有新的 tool call。
        """
        self._started_at = monotonic()
        self.status = TurnStatus.RUNNING
        if not isinstance(user_input.payload, UserMessagePayload):
            raise TypeError("turn requires a user message input")
        text = user_input.payload.text
        await self.session.emit(
            RuntimeEventType.TURN_STARTED,
            {"input_id": user_input.input_id},
            turn_id=self.turn_id,
        )
        await self.session.emit(
            RuntimeEventType.USER_MESSAGE,
            {"input_id": user_input.input_id, "text": text},
            turn_id=self.turn_id,
        )
        self.session.conversation.append_user_message(text)

        answer = ""

        while True:
            tool_calls: list[ToolCall] = []
            assistant_parts: list[str] = []
            completed_message: str | None = None
            response_completed = False

            model_messages = await self._prepare_model_messages()
            if model_messages is None:
                return

            request = ModelRequest(
                model=self.context.model,
                messages=tuple(model_messages),
                tools=tuple(self.context.available_tools),
            )
            async for model_event in self.session.llm_provider.stream(request):
                if model_event.type == ModelEventType.MESSAGE_DELTA:
                    if not isinstance(model_event.payload, ModelTextPayload):
                        raise LLMProtocolError("message delta has an invalid payload")
                    if completed_message is not None:
                        raise LLMProtocolError(
                            "message delta arrived after a completed message"
                        )
                    delta = model_event.payload.text
                    assistant_parts.append(delta)
                    await self.session.emit(
                        RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
                        {"text": delta},
                        turn_id=self.turn_id,
                    )

                elif model_event.type == ModelEventType.MESSAGE_COMPLETED:
                    if not isinstance(model_event.payload, ModelTextPayload):
                        raise LLMProtocolError(
                            "completed message has an invalid payload"
                        )
                    if assistant_parts or completed_message is not None:
                        raise LLMProtocolError(
                            "provider mixed streamed and completed message events"
                        )
                    completed_message = model_event.payload.text
                    await self.session.emit(
                        RuntimeEventType.ASSISTANT_MESSAGE,
                        {"text": completed_message},
                        turn_id=self.turn_id,
                    )
                    self.session.conversation.append_assistant_message(
                        completed_message
                    )

                elif model_event.type == ModelEventType.TOKEN_COUNT:
                    if not isinstance(model_event.payload, ModelTokenCountPayload):
                        raise LLMProtocolError("token count has an invalid payload")
                    await self.session.emit(
                        RuntimeEventType.TOKEN_COUNT,
                        model_event.payload.model_dump(mode="json"),
                        turn_id=self.turn_id,
                    )

                elif model_event.type == ModelEventType.TOOL_CALL:
                    if not isinstance(model_event.payload, ToolCall):
                        raise LLMProtocolError("tool call has an invalid payload")
                    tool_calls.append(model_event.payload)

                elif model_event.type == ModelEventType.COMPLETED:
                    response_completed = True
                    break

            if not response_completed:
                raise LLMProtocolError(
                    "model event stream ended without a completed event"
                )

            if tool_calls:
                await self._flush_streamed_message(
                    assistant_parts,
                    completed_message,
                )
                if self.tool_call_count + len(tool_calls) > self.context.max_tool_calls:
                    await self.abort(
                        "max_tool_calls_exceeded",
                        "Turn requested more tool calls than the configured limit.",
                        metadata={
                            "requested_tool_calls": [
                                call.model_dump(mode="json") for call in tool_calls
                            ],
                            "remaining_tool_calls": (
                                self.context.max_tool_calls - self.tool_call_count
                            ),
                        },
                    )
                    return
                await self._record_tool_calls(tool_calls)
                await self._run_tool_batch(tool_calls)
                continue

            if completed_message is None:
                completed_message = "".join(assistant_parts)
                if completed_message:
                    await self.session.emit(
                        RuntimeEventType.ASSISTANT_MESSAGE,
                        {"text": completed_message},
                        turn_id=self.turn_id,
                    )
                    self.session.conversation.append_assistant_message(
                        completed_message
                    )

            if not completed_message:
                raise LLMProtocolError(
                    "model completed without an assistant message or tool call"
                )
            answer = completed_message
            break

        await self.finish(answer)

    async def finish(self, answer: str) -> None:
        await self.session.emit(
            RuntimeEventType.TURN_FINISHED,
            {
                "answer": answer,
                "tool_calls": self.tool_call_count,
                "duration_ms": self._duration_ms(),
            },
            turn_id=self.turn_id,
        )
        self.status = TurnStatus.FINISHED

    async def _flush_streamed_message(
        self,
        assistant_parts: list[str],
        completed_message: str | None,
    ) -> str | None:
        if completed_message is not None:
            return completed_message

        streamed_message = "".join(assistant_parts)
        if not streamed_message:
            return None

        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE,
            {"text": streamed_message},
            turn_id=self.turn_id,
        )
        self.session.conversation.append_assistant_message(streamed_message)
        return streamed_message

    async def _record_tool_calls(self, calls: list[ToolCall]) -> None:
        for call in calls:
            await self.session.emit(
                RuntimeEventType.MODEL_TOOL_CALL,
                call.model_dump(mode="json"),
                turn_id=self.turn_id,
            )
        self.session.conversation.append_model_tool_calls(calls)

    async def _run_tool_batch(self, calls: list[ToolCall]) -> None:
        """Execute a provider batch and append results in provider order."""
        self.tool_call_count += len(calls)
        if self._can_parallelize(calls):
            semaphore = asyncio.Semaphore(self.context.max_parallel_read_tools)

            async def run(call: ToolCall) -> ToolResult:
                async with semaphore:
                    return await self._run_tool_call(call)

            results = await asyncio.gather(*(run(call) for call in calls))
        else:
            results = [await self._run_tool_call(call) for call in calls]

        for call, result in zip(calls, results, strict=True):
            self.session.conversation.append_tool_result(
                call.call_id,
                call.name,
                result.model_content(),
            )

    def _can_parallelize(self, calls: list[ToolCall]) -> bool:
        if len(calls) < 2 or self.context.max_parallel_read_tools < 2:
            return False
        for call in calls:
            try:
                tool = self.session.tool_registry.get(call.name)
            except CodecraftError:
                return False
            if tool.requires_approval or not tool.effects <= {ToolEffect.READ_ONLY}:
                return False
        return True

    async def _run_tool_call(self, call: ToolCall) -> ToolResult:
        """执行模型发起的 tool call，并把调用和结果写回 conversation。"""
        started_at = monotonic()
        result: ToolResult | None = None
        async for runner_event in self.session.tool_runner.run(call, self.context):
            await self.session.emit(
                runner_event.type,
                runner_event.payload,
                turn_id=self.turn_id,
            )
            if runner_event.type == RuntimeEventType.TOOL_CALL_FINISHED:
                result = ToolResult.model_validate(runner_event.payload["result"])

        if result is None:
            result = ToolResult(
                success=False,
                content="Tool did not produce a result.",
                error="tool_result_missing",
            )
            await self.session.emit(
                RuntimeEventType.TOOL_CALL_FINISHED,
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "result": result.model_dump(mode="json"),
                    "duration_ms": int((monotonic() - started_at) * 1000),
                },
                turn_id=self.turn_id,
            )

        return result

    async def _prepare_model_messages(self) -> list[ModelMessage] | None:
        messages = self.prompt_builder.build(
            config=self.session.config,
            conversation=self.session.conversation,
            context=self.context,
        )
        before_chars = self._model_input_chars(messages)
        if before_chars <= self.context.max_context_chars:
            return messages

        fixed_messages = self.prompt_builder.build(
            config=self.session.config,
            conversation=Conversation(),
            context=self.context,
        )
        history_budget = self.context.max_context_chars - self._model_input_chars(
            fixed_messages
        )
        compaction = None
        if history_budget > 0:
            compaction = self.session.conversation.compact(
                max_chars=history_budget,
                keep_recent_items=self.context.context_keep_recent_items,
            )

        if compaction is not None:
            await self.session.emit(
                RuntimeEventType.CONTEXT_COMPACTED,
                compaction,
                turn_id=self.turn_id,
            )
            messages = self.prompt_builder.build(
                config=self.session.config,
                conversation=self.session.conversation,
                context=self.context,
            )

        after_chars = self._model_input_chars(messages)
        if after_chars <= self.context.max_context_chars:
            return messages

        await self.abort(
            "context_limit_exceeded",
            "Model input exceeds the configured context budget.",
            metadata={
                "max_context_chars": self.context.max_context_chars,
                "input_chars": after_chars,
                "compaction_attempted": compaction is not None,
            },
        )
        return None

    def _model_input_chars(self, messages: list[ModelMessage]) -> int:
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [
                tool.model_dump(mode="json") for tool in self.context.available_tools
            ],
        }
        return len(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _build_context(self) -> TurnContext:
        config = self.session.config
        return TurnContext(
            session_id=config.session_id,
            turn_id=self.turn_id,
            cwd=config.cwd,
            workspace_roots=config.workspace_roots,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            sandbox_env_allowlist=config.sandbox_env_allowlist,
            available_tools=self.session.tool_registry.specs(),
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            turn_timeout_seconds=config.turn_timeout_seconds,
            tool_timeout_seconds=config.tool_timeout_seconds,
            approval_timeout_seconds=config.approval_timeout_seconds,
            max_context_chars=config.max_context_chars,
            context_keep_recent_items=config.context_keep_recent_items,
            max_parallel_read_tools=config.max_parallel_read_tools,
            created_at=datetime.now(UTC),
        )

    async def abort(
        self,
        reason: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.status in {TurnStatus.FINISHED, TurnStatus.ABORTED}:
            return
        await self.session.emit(
            RuntimeEventType.TURN_ABORTED,
            {
                "reason": reason,
                "message": message,
                "tool_calls": self.tool_call_count,
                "duration_ms": self._duration_ms(),
                "metadata": metadata or {},
            },
            turn_id=self.turn_id,
        )
        self.status = TurnStatus.ABORTED

    def _duration_ms(self) -> int:
        if self._started_at is None:
            return 0
        return int((monotonic() - self._started_at) * 1000)
