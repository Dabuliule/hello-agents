from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from codecraft.core.conversation import Conversation
from codecraft.core.errors import CodecraftError
from codecraft.core.token_budget import estimate_serialized_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.llm.base import LLMProtocolError
from codecraft.llm.base import ModelRequest
from codecraft.llm.events import (
    ModelEvent,
    ModelEventType,
    ModelTextPayload,
    ModelTokenCountPayload,
)
from codecraft.llm.messages import ModelMessage, ModelMessageType
from codecraft.prompt import InstructionLoader, PromptBuilder
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput, UserMessagePayload
from codecraft.schema.tool import ToolCall, ToolEffect, ToolResult
from codecraft.skill import Skill

if TYPE_CHECKING:
    from codecraft.core.session import Session


class TurnStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class _ModelResponse:
    assistant_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    completed_message: str | None = None


class Turn:
    """一次用户输入对应的模型执行轮次。

    `Turn` 管理从用户消息进入对话、调用模型、执行 tool call，到产出最终
    assistant 消息的完整循环。它不负责持久化细节，所有外部可见状态都通过
    session event 发出去。
    """

    _TOOL_RESULT_OVERHEAD_TOKENS = 64
    _MIN_TOOL_RESULT_TOKENS = 32

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
        self.instruction_loader = InstructionLoader()
        self._active_skills: dict[str, Skill] = {}
        self._started_at: float | None = None

    async def run(self, user_input: SessionInput) -> None:
        """运行一次用户输入，直到模型给出最终回复或轮次中止。

        模型可能在一次响应中要求调用工具；工具结果会回填到 conversation，
        然后继续下一次模型调用，直到没有新的 tool call。
        """
        await self._start(user_input)

        while True:
            model_messages = await self._prepare_model_messages()
            if model_messages is None:
                return

            request = ModelRequest(
                model=self.context.model,
                messages=tuple(model_messages),
                tools=tuple(self.context.available_tools),
                max_output_tokens=self.context.model_max_output_tokens,
            )
            response = await self._consume_model_response(request)

            if response.tool_calls:
                if not await self._process_tool_calls(response):
                    return
                continue

            answer = await self._final_answer(response)
            break

        await self.finish(answer)

    async def _start(self, user_input: SessionInput) -> None:
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
        for skill in self.session.skill_registry.explicit_mentions(text):
            self._active_skills[skill.metadata.name] = skill

    async def _consume_model_response(self, request: ModelRequest) -> _ModelResponse:
        response = _ModelResponse()
        async for model_event in self.session.llm_provider.stream(request):
            if model_event.type == ModelEventType.COMPLETED:
                return response
            await self._handle_model_event(response, model_event)
        raise LLMProtocolError("model event stream ended without a completed event")

    async def _handle_model_event(
        self,
        response: _ModelResponse,
        model_event: ModelEvent,
    ) -> None:
        if model_event.type == ModelEventType.MESSAGE_DELTA:
            await self._handle_message_delta(response, model_event)
        elif model_event.type == ModelEventType.MESSAGE_COMPLETED:
            await self._handle_completed_message(response, model_event)
        elif model_event.type == ModelEventType.TOKEN_COUNT:
            await self._handle_token_count(model_event)
        elif model_event.type == ModelEventType.TOOL_CALL:
            if not isinstance(model_event.payload, ToolCall):
                raise LLMProtocolError("tool call has an invalid payload")
            response.tool_calls.append(model_event.payload)

    async def _handle_message_delta(
        self,
        response: _ModelResponse,
        model_event: ModelEvent,
    ) -> None:
        if not isinstance(model_event.payload, ModelTextPayload):
            raise LLMProtocolError("message delta has an invalid payload")
        if response.completed_message is not None:
            raise LLMProtocolError("message delta arrived after a completed message")

        delta = model_event.payload.text
        response.assistant_parts.append(delta)
        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            {"text": delta},
            turn_id=self.turn_id,
        )

    async def _handle_completed_message(
        self,
        response: _ModelResponse,
        model_event: ModelEvent,
    ) -> None:
        if not isinstance(model_event.payload, ModelTextPayload):
            raise LLMProtocolError("completed message has an invalid payload")
        if response.assistant_parts or response.completed_message is not None:
            raise LLMProtocolError(
                "provider mixed streamed and completed message events"
            )

        response.completed_message = model_event.payload.text
        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE,
            {"text": response.completed_message},
            turn_id=self.turn_id,
        )
        self.session.conversation.append_assistant_message(response.completed_message)

    async def _handle_token_count(self, model_event: ModelEvent) -> None:
        if not isinstance(model_event.payload, ModelTokenCountPayload):
            raise LLMProtocolError("token count has an invalid payload")
        await self.session.emit(
            RuntimeEventType.TOKEN_COUNT,
            model_event.payload.model_dump(mode="json"),
            turn_id=self.turn_id,
        )

    async def _process_tool_calls(self, response: _ModelResponse) -> bool:
        await self._flush_streamed_message(
            response.assistant_parts,
            response.completed_message,
        )
        if (
            self.tool_call_count + len(response.tool_calls)
            > self.context.max_tool_calls
        ):
            await self._abort_for_tool_call_limit(response.tool_calls)
            return False
        await self._record_tool_calls(response.tool_calls)
        return await self._run_tool_batch(response.tool_calls)

    async def _abort_for_tool_call_limit(self, tool_calls: list[ToolCall]) -> None:
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

    async def _final_answer(self, response: _ModelResponse) -> str:
        completed_message = response.completed_message
        if completed_message is None:
            completed_message = "".join(response.assistant_parts)
            if completed_message:
                await self.session.emit(
                    RuntimeEventType.ASSISTANT_MESSAGE,
                    {"text": completed_message},
                    turn_id=self.turn_id,
                )
                self.session.conversation.append_assistant_message(completed_message)

        if not completed_message:
            raise LLMProtocolError(
                "model completed without an assistant message or tool call"
            )
        return completed_message

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

    async def _run_tool_batch(self, calls: list[ToolCall]) -> bool:
        """Execute a provider batch and append results in provider order."""
        self.tool_call_count += len(calls)
        output_tokens = await self._tool_result_token_limit(len(calls))
        if output_tokens is None:
            return False
        tool_context = self.context.model_copy(
            update={"max_tool_output_tokens": output_tokens}
        )
        if self._can_parallelize(calls):
            semaphore = asyncio.Semaphore(self.context.max_parallel_read_tools)

            async def run(call: ToolCall) -> ToolResult:
                async with semaphore:
                    return await self._run_tool_call(call, context=tool_context)

            results = await asyncio.gather(*(run(call) for call in calls))
        else:
            results = [
                await self._run_tool_call(call, context=tool_context) for call in calls
            ]

        for call, result in zip(calls, results, strict=True):
            self._activate_skill(call, result)
            self.session.conversation.append_tool_result(
                call.call_id,
                call.name,
                result.model_content(),
            )
        return True

    def _activate_skill(self, call: ToolCall, result: ToolResult) -> None:
        """在工具成功后激活 Skill；正文只进入下一轮 system prompt。"""
        if call.name != "load_skill" or not result.success:
            return
        name = call.arguments.get("name")
        if not isinstance(name, str):
            return
        self._active_skills[name] = self.session.skill_registry.get(name)

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

    async def _run_tool_call(
        self,
        call: ToolCall,
        *,
        context: TurnContext,
    ) -> ToolResult:
        """执行模型发起的 tool call，并把调用和结果写回 conversation。"""
        started_at = monotonic()
        result: ToolResult | None = None
        async for runner_event in self.session.tool_runner.run(call, context):
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
        project_instructions = self._project_instructions()
        messages = self._build_model_messages(project_instructions)
        input_budget = self._model_input_budget_tokens()
        before_tokens = self._model_input_tokens(messages)
        if before_tokens <= input_budget:
            return messages

        messages, compaction = self._compact_model_context(
            project_instructions=project_instructions,
            input_budget=input_budget,
        )
        if compaction is not None:
            await self.session.emit(
                RuntimeEventType.CONTEXT_COMPACTED,
                compaction,
                turn_id=self.turn_id,
            )

        after_tokens = self._model_input_tokens(messages)
        if after_tokens <= input_budget:
            return messages

        await self.abort(
            "context_limit_exceeded",
            "Model input exceeds the configured context budget.",
            metadata={
                "context_window_tokens": self.context.model_context_window_tokens,
                "input_budget_tokens": input_budget,
                "input_tokens": after_tokens,
                "compaction_attempted": compaction is not None,
            },
        )
        return None

    async def _tool_result_token_limit(self, call_count: int) -> int | None:
        project_instructions = self._project_instructions()
        messages = self._build_model_messages(project_instructions)
        input_budget = self._model_input_budget_tokens()
        current_tokens = self._model_input_tokens(messages)

        if current_tokens > input_budget:
            messages, compaction = self._compact_model_context(
                project_instructions=project_instructions,
                input_budget=input_budget,
            )
            if compaction is not None:
                await self.session.emit(
                    RuntimeEventType.CONTEXT_COMPACTED,
                    compaction,
                    turn_id=self.turn_id,
                )
            current_tokens = self._model_input_tokens(messages)

        overhead = self._TOOL_RESULT_OVERHEAD_TOKENS * call_count
        remaining = input_budget - current_tokens - overhead
        if remaining < self._MIN_TOOL_RESULT_TOKENS * call_count:
            await self.abort(
                "context_limit_exceeded",
                "Model input has no room for tool results and a follow-up response.",
                metadata={
                    "input_budget_tokens": input_budget,
                    "input_tokens": current_tokens,
                    "tool_calls": call_count,
                },
            )
            return None
        return min(self.context.max_tool_output_tokens, remaining // call_count)

    def _project_instructions(self) -> str | None:
        return self.instruction_loader.load_project_instructions(
            cwd=self.context.cwd,
            target_paths=self._instruction_target_paths(),
        )

    def _build_model_messages(
        self,
        project_instructions: str | None,
        *,
        conversation: Conversation | None = None,
    ) -> list[ModelMessage]:
        return self.prompt_builder.build(
            config=self.session.config,
            conversation=(
                conversation if conversation is not None else self.session.conversation
            ),
            context=self.context,
            project_instructions=project_instructions,
            available_skills=self.session.skill_registry.catalogue_prompt(),
            active_skills=self.session.skill_registry.active_prompt(
                self._active_skills.values()
            ),
        )

    def _compact_model_context(
        self,
        *,
        project_instructions: str | None,
        input_budget: int,
    ) -> tuple[list[ModelMessage], dict[str, Any] | None]:
        fixed_messages = self._build_model_messages(
            project_instructions,
            conversation=Conversation(),
        )
        history_budget = input_budget - self._model_input_tokens(fixed_messages)
        compaction = None
        if history_budget > 0:
            compaction = self.session.conversation.compact(
                max_tokens=history_budget,
                keep_recent_items=self.context.context_keep_recent_items,
            )
        return self._build_model_messages(project_instructions), compaction

    def _instruction_target_paths(self) -> list[Path]:
        """提取已访问路径，让后续模型调用获得对应目录的作用域指令。"""
        targets: list[Path] = []
        for item in self.session.conversation.items:
            if item.metadata.get("type") != ModelMessageType.TOOL_CALL.value:
                continue
            arguments = item.arguments or {}
            path = arguments.get("path")
            if isinstance(path, str) and path.strip():
                targets.append(Path(path))
        return targets

    def _model_input_tokens(self, messages: list[ModelMessage]) -> int:
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [
                tool.model_dump(mode="json") for tool in self.context.available_tools
            ],
        }
        return estimate_serialized_tokens(payload)

    def _model_input_budget_tokens(self) -> int:
        return (
            self.context.model_context_window_tokens
            - self.context.model_max_output_tokens
            - self.context.context_safety_margin_tokens
        )

    def _build_context(self) -> TurnContext:
        config = self.session.config
        return TurnContext(
            session_id=config.session_id,
            turn_id=self.turn_id,
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            sandbox_env_allowlist=config.sandbox_env_allowlist,
            available_tools=self.session.tool_registry.specs(),
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
