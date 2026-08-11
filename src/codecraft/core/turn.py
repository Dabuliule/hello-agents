from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, assert_never

from codecraft.core.conversation import Conversation, ConversationToolCallItem
from codecraft.core.errors import CodecraftError
from codecraft.core.token_budget import estimate_serialized_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.llm.base import LLMProtocolError
from codecraft.llm.base import ModelRequest
from codecraft.llm.events import (
    ModelCompletedEvent,
    ModelMessageCompletedEvent,
    ModelMessageDeltaEvent,
    ModelTokenCountEvent,
    ModelToolCallEvent,
)
from codecraft.llm.messages import ModelMessage
from codecraft.prompt import InstructionLoader, PromptBuilder
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput, UserMessagePayload
from codecraft.schema.tool import ToolCall, ToolEffect, ToolResult
from codecraft.skill import Skill

if TYPE_CHECKING:
    from codecraft.core.session import Session


class TurnStatus(StrEnum):
    """Turn 尚未开始、正在运行或已经成功/失败终止。"""

    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class _ModelResponse:
    """一次 Provider stream 中累积的增量文本、完整文本与 Tool calls。"""

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
    _MAX_SKILL_CATALOGUE_TOKENS = 2048
    _SKILL_CATALOGUE_BUDGET_DIVISOR = 8

    def __init__(
        self,
        *,
        session: Session,
        turn_id: str,
    ) -> None:
        """创建 TurnContext 快照和本轮独立的 Prompt/Instruction/Skill 状态。

        CREATED 允许 Session 先构造并公开 active_turn，再由后台 Task 调用 run；
        这让立即到来的 interrupt 仍有明确目标，也区分“对象存在”和“事件已开始”。
        """
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
        """进入 RUNNING，先持久化 Turn/User 事件，再追加历史和激活 mentions。"""
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
        """消费标准事件流直到显式 ModelCompleted；EOF 前缺终态视为协议错误。"""
        response = _ModelResponse()
        async for model_event in self.session.llm_provider.stream(request):
            if isinstance(model_event, ModelCompletedEvent):
                return response
            await self._handle_model_event(response, model_event)
        raise LLMProtocolError("model event stream ended without a completed event")

    async def _handle_model_event(
        self,
        response: _ModelResponse,
        model_event: ModelMessageDeltaEvent
        | ModelMessageCompletedEvent
        | ModelTokenCountEvent
        | ModelToolCallEvent,
    ) -> None:
        """穷举分发四类非终态 ModelEvent，未知联合成员触发类型不变量。"""
        if isinstance(model_event, ModelMessageDeltaEvent):
            await self._handle_message_delta(response, model_event)
        elif isinstance(model_event, ModelMessageCompletedEvent):
            await self._handle_completed_message(response, model_event)
        elif isinstance(model_event, ModelTokenCountEvent):
            await self._handle_token_count(model_event)
        elif isinstance(model_event, ModelToolCallEvent):
            response.tool_calls.append(model_event.payload)
        else:
            assert_never(model_event)

    async def _handle_message_delta(
        self,
        response: _ModelResponse,
        model_event: ModelMessageDeltaEvent,
    ) -> None:
        """累计文本 delta 并即时发 ASSISTANT_MESSAGE_DELTA；终态后禁止再流。"""
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
        model_event: ModelMessageCompletedEvent,
    ) -> None:
        """接收非流式完整消息，拒绝与 delta 混用并立即落盘 Conversation。"""
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

    async def _handle_token_count(self, model_event: ModelTokenCountEvent) -> None:
        """把 Provider 归一化 usage 原样持久化为 TOKEN_COUNT。"""
        await self.session.emit(
            RuntimeEventType.TOKEN_COUNT,
            model_event.payload.model_dump(mode="json"),
            turn_id=self.turn_id,
        )

    async def _process_tool_calls(self, response: _ModelResponse) -> bool:
        """先收口同响应文本，再检查总调用预算、记录 calls 并执行批次。"""
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
        """用请求明细和剩余额度中止超过 Turn 总工具数上限的整批 calls。"""
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
        """把流式片段合并成一次完整消息，拒绝无文本且无工具的成功终态。"""
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
        """持久化 answer、调用数和耗时后，将 Turn 标记 FINISHED。"""
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
        """Tool call 前把此前 delta 合成单条 Assistant 历史；完整事件不重复写。"""
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
        """逐个先持久化 MODEL_TOOL_CALL，再按 Provider 顺序批量追加历史。"""
        for call in calls:
            await self.session.emit(
                RuntimeEventType.MODEL_TOOL_CALL,
                call.model_dump(mode="json"),
                turn_id=self.turn_id,
            )
        self.session.conversation.append_model_tool_calls(calls)

    async def _run_tool_batch(self, calls: list[ToolCall]) -> bool:
        """执行同一 Provider 批次，并始终按 Provider 原顺序追加 ToolResults。

        批次开始即计入 max_tool_calls。全部工具共享按剩余上下文和 call 数均分
        的结果 Token 上限；只有至少两个、无审批且 effects 纯 READ_ONLY 的
        calls 才按 semaphore 并发，其他副作用调用严格串行。
        """
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
                """在只读批次并发上限内执行一个 call。"""
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
        """仅当所有工具存在、无需审批且 effects ⊆ READ_ONLY 时允许并发。"""
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
        """构造输入、必要时确定性压缩历史，仍超限则中止 Turn。

        固定输入先单独检查，因为 base/project/user/skills/context/tool schemas
        无法通过压缩 Conversation 消除；只有固定部分能装下才为历史分预算。
        """
        project_instructions = self._project_instructions()
        input_budget = self._model_input_budget_tokens()
        fixed_usage = await self._ensure_fixed_input_fits(
            project_instructions=project_instructions,
            input_budget=input_budget,
        )
        if fixed_usage is None:
            return None

        messages = self._build_model_messages(project_instructions)
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
                "fixed_input_tokens": fixed_usage["fixed_input_tokens"],
                "component_tokens": fixed_usage["component_tokens"],
            },
        )
        return None

    async def _tool_result_token_limit(self, call_count: int) -> int | None:
        """为本批每个 ToolResult 计算能容纳下一次模型请求的平均 Token 上限。

        先尝试压缩当前历史，再扣除每个 tool message 的协议 overhead；若连
        每项 32 Token 都放不下，直接中止而不是执行后丢弃 observation。
        """
        project_instructions = self._project_instructions()
        input_budget = self._model_input_budget_tokens()
        fixed_usage = await self._ensure_fixed_input_fits(
            project_instructions=project_instructions,
            input_budget=input_budget,
        )
        if fixed_usage is None:
            return None

        messages = self._build_model_messages(project_instructions)
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
                    "fixed_input_tokens": fixed_usage["fixed_input_tokens"],
                    "component_tokens": fixed_usage["component_tokens"],
                },
            )
            return None
        return min(self.context.max_tool_output_tokens, remaining // call_count)

    def _project_instructions(self) -> str | None:
        """按 cwd 与历史工具访问路径加载当前作用域的项目规则。"""
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
        """用固定 Prompt sections 与指定/当前 Conversation 构造模型消息。"""
        return self.prompt_builder.build(
            config=self.session.config,
            conversation=(
                conversation if conversation is not None else self.session.conversation
            ),
            context=self.context,
            project_instructions=project_instructions,
            available_skills=self._available_skills_prompt(),
            active_skills=self._active_skills_prompt(),
        )

    def _available_skills_prompt(self) -> str | None:
        """在独立 catalogue 预算内渲染轻量 Skill 目录。"""
        return self.session.skill_registry.catalogue_prompt(
            max_tokens=self._skill_catalogue_token_budget()
        )

    def _active_skills_prompt(self) -> str | None:
        """按首次激活顺序渲染本 Turn Skill 完整正文。"""
        return self.session.skill_registry.active_prompt(self._active_skills.values())

    def _skill_catalogue_token_budget(self) -> int:
        """取模型输入预算八分之一且最多 2048 Token 作为 Skill 目录上限。"""
        return min(
            self._MAX_SKILL_CATALOGUE_TOKENS,
            self._model_input_budget_tokens() // self._SKILL_CATALOGUE_BUDGET_DIVISOR,
        )

    async def _ensure_fixed_input_fits(
        self,
        *,
        project_instructions: str | None,
        input_budget: int,
    ) -> dict[str, Any] | None:
        """诊断固定输入是否能装入预算；不能时记录各 component 并中止。"""
        usage = self._fixed_input_usage(project_instructions)
        if usage["fixed_input_tokens"] <= input_budget:
            return usage

        await self.abort(
            "context_limit_exceeded",
            (
                "Fixed model input exceeds the configured context budget; reduce "
                "instructions, active Skills, or registered tool schemas."
            ),
            metadata={
                "detail": "fixed_input_exceeds_budget",
                "context_window_tokens": self.context.model_context_window_tokens,
                "input_budget_tokens": input_budget,
                **usage,
            },
        )
        return None

    def _fixed_input_usage(
        self,
        project_instructions: str | None,
    ) -> dict[str, Any]:
        """估算空历史 Prompt sections、Tool schemas 和 Skill 状态的 Token 组成。"""
        available_skills = self._available_skills_prompt()
        active_skills = self._active_skills_prompt()
        fixed_messages = self.prompt_builder.build(
            config=self.session.config,
            conversation=Conversation(),
            context=self.context,
            project_instructions=project_instructions,
            available_skills=available_skills,
            active_skills=active_skills,
        )
        component_tokens = self.prompt_builder.fixed_section_tokens(
            config=self.session.config,
            context=self.context,
            project_instructions=project_instructions,
            available_skills=available_skills,
            active_skills=active_skills,
        )
        component_tokens["tool_schemas"] = estimate_serialized_tokens(
            [tool.model_dump(mode="json") for tool in self.context.available_tools]
        )
        return {
            "fixed_input_tokens": self._model_input_tokens(fixed_messages),
            "component_tokens": component_tokens,
            "skill_catalogue_budget_tokens": self._skill_catalogue_token_budget(),
            "active_skills": list(self._active_skills),
            "tool_count": len(self.context.available_tools),
        }

    def _compact_model_context(
        self,
        *,
        project_instructions: str | None,
        input_budget: int,
    ) -> tuple[list[ModelMessage], dict[str, Any] | None]:
        """从输入总预算减固定消息，为 Conversation 执行一次原地压缩。"""
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
            if not isinstance(item, ConversationToolCallItem):
                continue
            path = item.arguments.get("path")
            if isinstance(path, str) and path.strip():
                targets.append(Path(path))
        return targets

    def _model_input_tokens(self, messages: list[ModelMessage]) -> int:
        """按 Provider 请求真实 messages+tools JSON 形态估算输入 Token。"""
        payload = {
            "messages": [message.model_dump(mode="json") for message in messages],
            "tools": [
                tool.model_dump(mode="json") for tool in self.context.available_tools
            ],
        }
        return estimate_serialized_tokens(payload)

    def _model_input_budget_tokens(self) -> int:
        """上下文窗口扣除最大输出和安全余量，得到每次请求输入预算。

        Example:
            131072 窗口、8192 最大输出、2048 safety margin 对应 120832
            个估算输入 Token。
        """
        return (
            self.context.model_context_window_tokens
            - self.context.model_max_output_tokens
            - self.context.context_safety_margin_tokens
        )

    def _build_context(self) -> TurnContext:
        """从 SessionConfig 和当时 Tool specs 构造不可变 TurnContext 快照。"""
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
        """幂等持久化 TURN_ABORTED 的原因、调用数、耗时和诊断 metadata。"""
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
        """返回从 _start 起的单调时钟毫秒数，尚未开始返回 0。"""
        if self._started_at is None:
            return 0
        return int((monotonic() - self._started_at) * 1000)
