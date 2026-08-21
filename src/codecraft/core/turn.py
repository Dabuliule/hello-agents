from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, assert_never

from codecraft.core.async_utils import finish_task_before_cancelling
from codecraft.core.conversation import Conversation, ConversationToolCallItem
from codecraft.core.errors import CodecraftError
from codecraft.core.token_budget import estimate_serialized_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.llm.base import LLMProtocolError
from codecraft.llm.base import ModelRequest
from codecraft.llm.events import (
    ModelCompletedEvent,
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
    """一次已成功闭合 Provider stream 的文本与 Tool calls 临时累加器。

    Token usage 会立即转成 RuntimeEvent，不参与后续分支判断，因此不保存在这里。
    所有 Provider 文本都表示为一个或多个有序 parts，等确认整个响应成功后再合并
    进 Conversation；非流式上游响应由 adapter 转换为单个 part。
    """

    assistant_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class _ToolCallState:
    """跟踪已记录 call 的运行边界，供异常收口时补齐 ToolResult。"""

    started_at: float | None = None
    result: ToolResult | None = None


class Turn:
    """一次用户输入对应的模型执行轮次。

    一个 Turn 可以包含多次模型请求：模型要求工具时，Turn 先执行并回填结果，
    再携带更新后的 Conversation 请求模型，直到得到无 tool call 的最终回复。
    它负责这条业务循环，不负责后台 Task 的 deadline、取消和状态清理；后者由
    ``Session._run_turn`` 托管。所有外部可见状态都通过 Session event 发出。
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

        Args:
            user_input: 本 Turn 唯一的 USER_MESSAGE；审批和中止走 Session 控制面。

        模型可能在一次响应中要求调用工具；工具结果会回填到 Conversation，
        然后 ``continue`` 发起下一次模型请求。准备上下文或工具阶段主动中止时，
        helper 已记录 ``TURN_ABORTED`` 并以 False/None 通知本方法直接返回；其他
        异常继续抛给 ``Session._run_turn`` 统一转成 ERROR/终态和清理 Session。

        状态机可以概括为 ``USER → MODEL → (TOOLS → MODEL)* → FINAL``。
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
        """消费一次 Provider 事件流，只在显式成功终态后返回累积响应。

        Args:
            request: 由当前 Prompt、Tool specs 和输出预算构成的不可变请求。

        Returns:
            已收集文本和 ToolCall 的 ``_ModelResponse``。ToolCall 在整个响应确认
            完成前不会执行，避免使用提前断流留下的不完整意图。

        Raises:
            LLMProtocolError: 迭代器在 ``ModelCompletedEvent`` 前结束。普通 EOF
                不能当成功，因为网络断流与正常闭合在传输层可能表现相同。
            ModelProviderError: Provider 报告的配置、网络或上游协议错误会原样上抛，
                再由 ``Session._run_turn`` 转成 Runtime ERROR/终态事件。
        """
        response = _ModelResponse()
        async for model_event in self.session.llm_provider.stream(request):
            if isinstance(model_event, ModelCompletedEvent):
                return response
            await self._handle_model_event(response, model_event)
        raise LLMProtocolError("model event stream ended without a completed event")

    async def _handle_model_event(
        self,
        response: _ModelResponse,
        model_event: ModelMessageDeltaEvent | ModelTokenCountEvent | ModelToolCallEvent,
    ) -> None:
        """穷举处理三类非终态事件，并维持文本协议不变量。

        文本 part 会立即广播以支持 UI 显示，同时在本地累积；usage 只发审计事件；
        ToolCall 等待 stream 闭合后批量执行。``ModelCompletedEvent`` 已由外层循环
        处理，不进入本方法。
        """
        if isinstance(model_event, ModelMessageDeltaEvent):
            await self._handle_message_delta(response, model_event)
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
        """累计一个有序文本 part，并即时发 ASSISTANT_MESSAGE_DELTA。"""
        delta = model_event.payload.text
        response.assistant_parts.append(delta)
        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            {"text": delta},
            turn_id=self.turn_id,
        )

    async def _handle_token_count(self, model_event: ModelTokenCountEvent) -> None:
        """把 Provider 归一化 usage 原样持久化为 TOKEN_COUNT。"""
        await self.session.emit(
            RuntimeEventType.TOKEN_COUNT,
            model_event.payload.model_dump(mode="json"),
            turn_id=self.turn_id,
        )

    async def _process_tool_calls(self, response: _ModelResponse) -> bool:
        """收口响应文本，并把整批 calls 作为完整协议单元记录、执行。

        ``max_tool_calls`` 是跨多次模型请求的 Turn 累计上限，而不是单响应上限。
        超限时整批都不记录、不执行，避免只执行模型同批请求的一部分而破坏其并行
        意图；预算也先在 Conversation 副本中预演，通过后才记录 MODEL_TOOL_CALL。

        一旦开始记录，取消或内部异常都必须为整批补齐 TOOL_CALL_FINISHED 和历史
        ToolResult。这样实时 Conversation 与事件重建都不会留下只有 call、没有
        observation 的非法 Provider 上下文。
        """
        await self._flush_streamed_message(response.assistant_parts)
        if (
            self.tool_call_count + len(response.tool_calls)
            > self.context.max_tool_calls
        ):
            await self._abort_for_tool_call_limit(response.tool_calls)
            return False

        output_tokens = await self._tool_result_token_limit(response.tool_calls)
        if output_tokens is None:
            return False

        calls = response.tool_calls
        states = [_ToolCallState() for _ in calls]
        self.tool_call_count += len(calls)
        recorded = False
        try:
            await self._record_tool_calls(calls)
            recorded = True
            results = await self._run_tool_batch(
                calls,
                states=states,
                output_tokens=output_tokens,
            )
        except asyncio.CancelledError as exc:
            results = await self._complete_incomplete_tool_calls(
                calls,
                states,
                error="tool_interrupted",
                reason=str(exc.args[0]) if exc.args else "turn_cancelled",
            )
            self._append_tool_results(calls, results)
            raise
        except Exception as exc:
            if not recorded:
                raise
            results = await self._complete_incomplete_tool_calls(
                calls,
                states,
                error="tool_runtime_interrupted",
                reason=type(exc).__name__,
            )
            self._append_tool_results(calls, results)
            raise

        self._append_tool_results(calls, results)
        return True

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
        """把文本 parts 合并并持久化，拒绝无文本且无工具的成功终态。"""
        answer = "".join(response.assistant_parts)
        if not answer:
            raise LLMProtocolError(
                "model completed without an assistant message or tool call"
            )
        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE,
            {"text": answer},
            turn_id=self.turn_id,
        )
        self.session.conversation.append_assistant_message(answer)
        return answer

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
    ) -> str | None:
        """Tool call 前把本响应文本 parts 合成单条 Assistant 历史。"""
        assistant_message = "".join(assistant_parts)
        if not assistant_message:
            return None

        await self.session.emit(
            RuntimeEventType.ASSISTANT_MESSAGE,
            {"text": assistant_message},
            turn_id=self.turn_id,
        )
        self.session.conversation.append_assistant_message(assistant_message)
        return assistant_message

    async def _record_tool_calls(self, calls: list[ToolCall]) -> None:
        """不可分割地持久化整批 MODEL_TOOL_CALL，再批量追加实时历史。

        记录在独立 Task 中完成；若调用方此时被取消，先完成整批记录再传播取消，
        让 ``_process_tool_calls`` 能为同一批 calls 统一补齐失败 ToolResult。
        """

        async def record() -> None:
            """按 Provider 顺序写完事件和对应的 Conversation calls。"""
            for call in calls:
                await self.session.emit(
                    RuntimeEventType.MODEL_TOOL_CALL,
                    call.model_dump(mode="json"),
                    turn_id=self.turn_id,
                )
            self.session.conversation.append_model_tool_calls(calls)

        task = asyncio.create_task(record())
        await finish_task_before_cancelling(task)

    async def _run_tool_batch(
        self,
        calls: list[ToolCall],
        *,
        states: list[_ToolCallState],
        output_tokens: int,
    ) -> list[ToolResult]:
        """执行已记录批次，返回与 Provider calls 一一对应的结果。

        Args:
            calls: 已通过 Turn 总量和上下文预算检查、写入历史的调用列表。
            states: 与 calls 同序的执行状态；异常收口依靠它区分已有结果和缺口。
            output_tokens: 预算预演得到的单个 ToolResult 最大模型可见 Token 数。

        并发批次收到取消或内部异常时，会先取消并等待所有子 Task，确保后续补写
        结果时不会再有后台工具产生重复的 TOOL_CALL_FINISHED。
        """
        tool_context = self.context.model_copy(
            update={"max_tool_output_tokens": output_tokens}
        )
        if self._can_parallelize(calls):
            semaphore = asyncio.Semaphore(self.context.max_parallel_read_tools)

            async def run(
                call: ToolCall,
                state: _ToolCallState,
            ) -> ToolResult:
                """在只读批次并发上限内执行一个 call。"""
                async with semaphore:
                    return await self._run_tool_call(
                        call,
                        context=tool_context,
                        state=state,
                    )

            tasks = [
                asyncio.create_task(run(call, state))
                for call, state in zip(calls, states, strict=True)
            ]
            try:
                # gather 的执行完成顺序可以不同，但返回列表保持 awaitable 输入顺序；
                # Conversation 因而仍按 Provider call 顺序构造工具历史。
                return await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        return [
            await self._run_tool_call(call, context=tool_context, state=state)
            for call, state in zip(calls, states, strict=True)
        ]

    async def _complete_incomplete_tool_calls(
        self,
        calls: list[ToolCall],
        states: list[_ToolCallState],
        *,
        error: str,
        reason: str,
    ) -> list[ToolResult]:
        """为异常退出时尚无终态的 calls 生成并持久化保守失败结果。

        已见 TOOL_CALL_STARTED 的调用可能已经产生外部副作用，因此标记
        ``outcome_unknown=True`` 且禁止自动重试；未开始的调用可安全重试。
        """
        results: list[ToolResult] = []
        for call, state in zip(calls, states, strict=True):
            if state.result is not None:
                results.append(state.result)
                continue

            started = state.started_at is not None
            result_error = error if started else "tool_not_started"
            result = ToolResult(
                success=False,
                content=(
                    "Tool processing was interrupted before a result was available."
                    if started
                    else "Tool was not started because the turn stopped."
                ),
                error=result_error,
                metadata={
                    "outcome_unknown": started,
                    "retry_safe": not started,
                    "interruption_reason": reason,
                },
            )
            state.result = result
            await self.session.emit(
                RuntimeEventType.TOOL_CALL_FINISHED,
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "result": result.model_dump(mode="json"),
                    "duration_ms": (
                        int((monotonic() - state.started_at) * 1000)
                        if state.started_at is not None
                        else 0
                    ),
                },
                turn_id=self.turn_id,
            )
            results.append(result)
        return results

    def _append_tool_results(
        self,
        calls: list[ToolCall],
        results: list[ToolResult],
    ) -> None:
        """按 Provider 原顺序激活 Skill 并追加一一对应的 ToolResults。"""
        for call, result in zip(calls, results, strict=True):
            self._activate_skill(call, result)
            self.session.conversation.append_tool_result(
                call.call_id,
                call.name,
                result.model_content(),
            )

    def _activate_skill(self, call: ToolCall, result: ToolResult) -> None:
        """在工具成功后激活 Skill；正文只进入下一轮 system prompt。"""
        if call.name != "load_skill" or not result.success:
            return
        name = call.arguments.get("name")
        if not isinstance(name, str):
            return
        self._active_skills[name] = self.session.skill_registry.get(name)

    def _can_parallelize(self, calls: list[ToolCall]) -> bool:
        """仅当整批工具存在、无需审批且 effects ⊆ READ_ONLY 时允许并发。

        任何未知工具或混入的副作用工具都会让整个批次 fail-closed 为串行；不能只
        并发其中的只读子集，否则事件和副作用顺序将偏离模型给出的批次结构。
        """
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
        state: _ToolCallState,
    ) -> ToolResult:
        """转发单次 ToolRunner 事件并返回终态结果，不在此处修改 Conversation。

        批次结果全部返回后，``_run_tool_batch`` 才按 Provider 原顺序写入历史；并发
        工具的 RuntimeEvent 允许按真实完成时间交错，依靠 call_id 关联各自调用。
        """
        started_at = monotonic()
        async for runner_event in self.session.tool_runner.run(call, context):
            if runner_event.type == RuntimeEventType.TOOL_CALL_STARTED:
                state.started_at = started_at
            elif runner_event.type == RuntimeEventType.TOOL_CALL_FINISHED:
                state.result = ToolResult.model_validate(runner_event.payload["result"])
            await self.session.emit(
                runner_event.type,
                runner_event.payload,
                turn_id=self.turn_id,
            )

        if state.result is None:
            state.result = ToolResult(
                success=False,
                content="Tool did not produce a result.",
                error="tool_result_missing",
            )
            await self.session.emit(
                RuntimeEventType.TOOL_CALL_FINISHED,
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "result": state.result.model_dump(mode="json"),
                    "duration_ms": int((monotonic() - started_at) * 1000),
                },
                turn_id=self.turn_id,
            )

        return state.result

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

    async def _tool_result_token_limit(
        self,
        calls: list[ToolCall],
    ) -> int | None:
        """在不写入 calls 的前提下预演并分配每项 ToolResult Token 上限。

        临时 Conversation 包含候选 calls，因此参数体和新路径指令也计入下一轮
        模型输入。若空间不足，按 calls 加最小结果空间反推当前历史预算并压缩真实
        Conversation；压缩快照本身不含尚未接受的 calls。仍放不下时直接中止，
        不产生 MODEL_TOOL_CALL，更不会执行工具后丢弃 observation。
        """
        call_count = len(calls)
        project_instructions = self._project_instructions(tool_calls=calls)
        input_budget = self._model_input_budget_tokens()
        fixed_usage = await self._ensure_fixed_input_fits(
            project_instructions=project_instructions,
            input_budget=input_budget,
        )
        if fixed_usage is None:
            return None

        current_messages = self._build_model_messages(project_instructions)
        current_tokens = self._model_input_tokens(current_messages)
        prospective = self.session.conversation.model_copy(deep=True)
        prospective.append_model_tool_calls(calls)
        messages = self._build_model_messages(
            project_instructions,
            conversation=prospective,
        )
        prospective_tokens = self._model_input_tokens(messages)
        result_reserve = (
            self._TOOL_RESULT_OVERHEAD_TOKENS + self._MIN_TOOL_RESULT_TOKENS
        ) * call_count

        if prospective_tokens + result_reserve > input_budget:
            call_tokens = max(0, prospective_tokens - current_tokens)
            _, compaction = self._compact_model_context(
                project_instructions=project_instructions,
                input_budget=input_budget - call_tokens - result_reserve,
            )
            if compaction is not None:
                await self.session.emit(
                    RuntimeEventType.CONTEXT_COMPACTED,
                    compaction,
                    turn_id=self.turn_id,
                )
            prospective = self.session.conversation.model_copy(deep=True)
            prospective.append_model_tool_calls(calls)
            messages = self._build_model_messages(
                project_instructions,
                conversation=prospective,
            )
            prospective_tokens = self._model_input_tokens(messages)

        overhead = self._TOOL_RESULT_OVERHEAD_TOKENS * call_count
        remaining = input_budget - prospective_tokens - overhead
        if remaining < self._MIN_TOOL_RESULT_TOKENS * call_count:
            await self.abort(
                "context_limit_exceeded",
                "Model input has no room for tool results and a follow-up response.",
                metadata={
                    "input_budget_tokens": input_budget,
                    "input_tokens": prospective_tokens,
                    "tool_calls": call_count,
                    "requested_tool_calls": [
                        call.model_dump(mode="json") for call in calls
                    ],
                    "fixed_input_tokens": fixed_usage["fixed_input_tokens"],
                    "component_tokens": fixed_usage["component_tokens"],
                },
            )
            return None
        return min(self.context.max_tool_output_tokens, remaining // call_count)

    def _project_instructions(
        self,
        *,
        tool_calls: list[ToolCall] | None = None,
    ) -> str | None:
        """按 cwd、历史访问路径和可选候选 calls 加载作用域项目规则。"""
        targets = self._instruction_target_paths()
        for call in tool_calls or []:
            path = call.arguments.get("path")
            if isinstance(path, str) and path.strip():
                targets.append(Path(path))
        return self.instruction_loader.load_project_instructions(
            cwd=self.context.cwd,
            target_paths=targets,
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
        """从 SessionConfig 和当前 Tool specs 构造本 Turn 的最小权限快照。

        Context 只下发模型标识、workspace、审批/沙箱规则、可见工具和执行预算，
        不把 Provider 密钥变量、SessionStore 或 Runtime 资源交给 ToolRunner。工具
        目录在 Turn 创建时截取，保证本轮 Prompt 和执行治理使用同一组 specs。
        """
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
