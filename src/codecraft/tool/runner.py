from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
import json
from time import monotonic
from typing import Any

from pydantic import ValidationError

from codecraft.approval.manager import (
    ApprovalDecision,
    ApprovalEvaluation,
    ApprovalManager,
    ApprovalRequest,
)
from codecraft.core.errors import CodecraftError
from codecraft.core.token_budget import estimate_text_tokens, truncate_text_to_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox.policy import SandboxPolicy
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.tool import ToolCall, ToolResult
from codecraft.tool.base import ToolContext
from codecraft.tool.observer import ToolResultObserver
from codecraft.tool.registry import ToolRegistry


@dataclass(frozen=True)
class ToolRunnerEvent:
    """ToolRunner 向 Turn 产生的标准 RuntimeEvent 类型与 payload。"""

    type: RuntimeEventType
    payload: dict[str, Any]


@dataclass
class _ToolRunState:
    """跨异步生成器阶段共享的结果、审批和分段计时状态。"""

    started_at: float
    result: ToolResult | None = None
    approved: bool = False
    approval_wait_ms: int = 0
    execution_ms: int = 0
    observer_ms: int = 0
    execution_deadline: asyncio.Timeout | None = None


@dataclass(frozen=True)
class _ApprovalOutcome:
    """审批决定、等待耗时和 reviewer 失败诊断。"""

    decision: ApprovalDecision
    wait_ms: int
    error: str | None = None
    exception_type: str | None = None


class ToolRunner:
    """统一执行 tool call，并把执行过程转成 RuntimeEvent。

    Runner 是模型意图与真实副作用之间的治理边界，固定顺序为：Registry 查找、
    参数 schema 校验、Sandbox effect 检查、Approval 检查、限时执行、成功结果
    Observer、输出限额。Sandbox 是不可被审批扩大的硬能力边界；Approval 只决定
    已在能力范围内的本次操作是否获得用户授权。

    参数、策略、审批、工具超时和普通执行异常都归一化为失败 ToolResult，让模型
    可以观察失败后修正参数或解释原因，而不是直接终止整个 Turn。任务取消不在这里
    吞掉：``CancelledError`` 必须向上传播，由 Turn 为已记录批次补齐终态，再由
    Session 记录 interrupt/turn timeout。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        approval_manager: ApprovalManager | None = None,
        observers: Sequence[ToolResultObserver] | None = None,
    ) -> None:
        """绑定工具/审批 Registry 与名称唯一的非关键结果 Observer。"""
        self.registry = registry
        self.approval_manager = approval_manager or ApprovalManager()
        self.observers = tuple(observers or ())
        names = [observer.name for observer in self.observers]
        if len(names) != len(set(names)):
            raise ValueError("tool result observers must have unique names")

    async def run(
        self,
        call: ToolCall,
        context: TurnContext,
    ) -> AsyncIterator[ToolRunnerEvent]:
        """运行一个 tool call，并按开始、审批、结束、附加事件顺序产出事件。

        Args:
            call: 模型给出的工具名、call_id 与尚未校验的 arguments。
            context: 当前 Turn 冻结的权限、预算、超时和 workspace 快照。

        Yields:
            正常消费至结束时必有 TOOL_CALL_STARTED/FINISHED；需审批时夹有
            REQUESTED/DECIDED；工具声明的 runtime_events 在 FINISHED 之后且
            payload 被限额。外部取消会中断生成器，缺失终态由 Turn 批次收口补齐。

        所有预期参数、业务、审批、超时和普通执行异常都变成 ToolResult，使
        Turn 能把失败 observation 交还模型；只有内部“不产结果”不变量会抛出。
        """
        yield ToolRunnerEvent(
            RuntimeEventType.TOOL_CALL_STARTED,
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            },
        )

        state = _ToolRunState(started_at=monotonic())
        async for event in self._run_pipeline(call, context, state):
            yield event

        result = state.result
        if result is None:  # pragma: no cover - internal pipeline invariant
            raise RuntimeError("tool execution pipeline produced no result")

        yield ToolRunnerEvent(
            RuntimeEventType.TOOL_CALL_FINISHED,
            self._finished_payload(
                call,
                result,
                started_at=state.started_at,
                approval_wait_ms=state.approval_wait_ms,
                execution_ms=state.execution_ms,
                observer_ms=state.observer_ms,
            ),
        )

        for runtime_event in result.runtime_events:
            yield ToolRunnerEvent(
                runtime_event.type,
                {
                    **self._limit_mapping(
                        runtime_event.payload,
                        context.max_tool_output_chars,
                        label="payload",
                    ),
                    "call_id": call.call_id,
                },
            )

    async def _run_pipeline(
        self,
        call: ToolCall,
        context: TurnContext,
        state: _ToolRunState,
    ) -> AsyncIterator[ToolRunnerEvent]:
        """把可恢复治理/执行失败收口为 ToolResult，再治理成功结果和输出。

        ``ValidationError``、领域 ``CodecraftError``、Runner 自己的执行 deadline
        和未知普通异常使用稳定错误码；``CancelledError`` 继承 BaseException，故
        不会被最后的 ``except Exception`` 误包装成普通工具失败。

        Observer 是非关键后处理，只在原工具成功后运行；其失败只进入诊断，不会
        倒置已经发生的工具成功事实。所有路径最后都经过统一输出限额。
        """
        try:
            async for event in self._prepare_and_execute(call, context, state):
                yield event
        except TimeoutError as exc:
            state.result = self._timeout_result(exc, context, state)
        except ValidationError as exc:
            state.result = self._validation_result(exc)
        except CodecraftError as exc:
            state.result = ToolResult(
                success=False,
                content=exc.message,
                error=exc.code,
                suggestion=exc.suggestion,
                metadata=exc.metadata,
            )
        except Exception as exc:
            state.result = ToolResult(
                success=False,
                content="Tool execution failed.",
                error="tool_execution_error",
                suggestion="Check the tool arguments, workspace permissions, or runtime environment.",
                metadata={"exception_type": type(exc).__name__},
            )

        result = state.result
        if result is None:  # pragma: no cover - internal pipeline invariant
            raise RuntimeError("tool execution stage produced no result")

        if result.success and self.observers:
            observer_started_at = monotonic()
            post_actions = await self._run_observers(call, result, context)
            state.observer_ms = int((monotonic() - observer_started_at) * 1000)
            if post_actions:
                result.metadata["post_actions"] = post_actions

        state.result = self._limit_output(
            result,
            context.max_tool_output_chars,
            context.max_tool_output_tokens,
        )

    async def _prepare_and_execute(
        self,
        call: ToolCall,
        context: TurnContext,
        state: _ToolRunState,
    ) -> AsyncIterator[ToolRunnerEvent]:
        """依次取工具、校验参数、检查 Sandbox、审批并在 deadline 内执行。

        Sandbox 是审批无法覆盖的硬边界，因此 effect 不允许时直接形成
        ``sandbox_denied``，既不创建 ApprovalRequest，也不调用工具。审批只处理
        Sandbox 已允许、但当前 policy 或命令风险要求用户确认的操作。

        ``ToolContext`` 只把冻结的 TurnContext、当前 call、审批结果和命令分类交给
        工具，不暴露 Registry、Reviewer 或 Session；Bash 等高风险工具会再次检查
        ``command_decision`` 和 ``approved``，形成执行边界处的纵深防御。
        """
        tool = self.registry.get(call.name)
        args = tool.args_schema.model_validate(call.arguments)
        sandbox_evaluation = self._sandbox_policy(context).evaluate_effects(
            tool.effects
        )
        if not sandbox_evaluation.allowed:
            # sandbox 是硬边界；不进入 approval，也不执行 tool。
            state.result = ToolResult(
                success=False,
                content="Tool execution denied by sandbox policy.",
                error="sandbox_denied",
                suggestion=sandbox_evaluation.reason,
                metadata={
                    "tool": call.name,
                    "sandbox_mode": context.sandbox_mode,
                    "denied_effect": sandbox_evaluation.denied_effect,
                },
            )
            return

        evaluation = await self.approval_manager.evaluate(tool, call, args, context)
        if evaluation.requires_approval:
            async for event in self._handle_approval(call, context, evaluation, state):
                yield event
            if state.result is not None:
                return

        execution_started_at = monotonic()
        state.execution_deadline = asyncio.timeout(context.tool_timeout_seconds)
        try:
            async with state.execution_deadline:
                state.result = await tool.arun(
                    args,
                    ToolContext(
                        context=context,
                        call=call,
                        approved=state.approved,
                        command_decision=evaluation.command_decision,
                    ),
                )
        finally:
            state.execution_ms = int((monotonic() - execution_started_at) * 1000)

    async def _handle_approval(
        self,
        call: ToolCall,
        context: TurnContext,
        evaluation: ApprovalEvaluation,
        state: _ToolRunState,
    ) -> AsyncIterator[ToolRunnerEvent]:
        """按 REQUESTED → 等待 Reviewer → DECIDED 的可审计顺序处理审批。

        同意只把 ``state.approved`` 置 True，真正执行仍回到统一工具路径；拒绝、
        Reviewer 异常和审批超时都形成失败 ToolResult。任务取消继续向上传播，
        ThreadApprovalReviewer 的 finally 会同步清理 pending Future。
        """
        # approval 是可交互边界，先产出请求，再等待 UI 或 reviewer 处理。
        request = self.approval_manager.build_request(
            call=call,
            context=context,
            evaluation=evaluation,
        )
        yield ToolRunnerEvent(
            RuntimeEventType.APPROVAL_REQUESTED,
            request.model_dump(mode="json"),
        )

        outcome = await self._review_approval(
            request,
            timeout_seconds=context.approval_timeout_seconds,
        )
        state.approval_wait_ms = outcome.wait_ms
        yield ToolRunnerEvent(
            RuntimeEventType.APPROVAL_DECIDED,
            outcome.decision.model_dump(mode="json"),
        )
        if outcome.decision.approved:
            state.approved = True
            return

        state.result = self._approval_denied_result(call, outcome)

    async def _review_approval(
        self,
        request: ApprovalRequest,
        *,
        timeout_seconds: int,
    ) -> _ApprovalOutcome:
        """限时调用 Reviewer，并把超时/异常转换成 fail-closed 拒绝决定。"""
        started_at = monotonic()
        error: str | None = None
        exception_type: str | None = None
        deadline = asyncio.timeout(timeout_seconds)
        try:
            async with deadline:
                decision = await self.approval_manager.request(request)
        except TimeoutError as exc:
            timed_out = deadline.expired()
            error = "approval_timeout" if timed_out else "approval_error"
            exception_type = type(exc).__name__
            decision = self.approval_manager.build_reviewer_failure_decision(
                request,
                timed_out=timed_out,
            )
        except Exception as exc:
            error = "approval_error"
            exception_type = type(exc).__name__
            decision = self.approval_manager.build_reviewer_failure_decision(
                request,
                timed_out=False,
            )
        finally:
            wait_ms = int((monotonic() - started_at) * 1000)

        return _ApprovalOutcome(
            decision=decision,
            wait_ms=wait_ms,
            error=error,
            exception_type=exception_type,
        )

    @staticmethod
    def _approval_denied_result(
        call: ToolCall,
        outcome: _ApprovalOutcome,
    ) -> ToolResult:
        """把用户拒绝、Reviewer 错误或审批超时映射为稳定 ToolResult。"""
        metadata = {
            "approval_id": outcome.decision.approval_id,
            "tool": call.name,
        }
        if outcome.exception_type is not None:
            metadata["exception_type"] = outcome.exception_type

        return ToolResult(
            success=False,
            content=(
                "Tool approval timed out."
                if outcome.error == "approval_timeout"
                else "Tool execution denied by approval."
            ),
            error=outcome.error or "approval_denied",
            suggestion=outcome.decision.reason,
            metadata=metadata,
        )

    @staticmethod
    def _timeout_result(
        exc: TimeoutError,
        context: TurnContext,
        state: _ToolRunState,
    ) -> ToolResult:
        """只把 Runner 自己的 execution deadline 过期标成 outcome_unknown。

        工具内部抛出的 TimeoutError 不是 Runner 超时，按普通 execution_error
        处理。真正超时时操作可能已产生外部副作用，因此 retry_safe=False。
        """
        if state.execution_deadline is not None and state.execution_deadline.expired():
            return ToolResult(
                success=False,
                content="Tool execution timed out.",
                error="tool_timeout",
                suggestion=(
                    "The operation may still have completed; inspect state before "
                    "retrying or increasing the timeout."
                ),
                metadata={
                    "timeout_seconds": context.tool_timeout_seconds,
                    "outcome_unknown": True,
                    "retry_safe": False,
                },
            )
        return ToolResult(
            success=False,
            content="Tool execution failed.",
            error="tool_execution_error",
            suggestion="Check the tool arguments or runtime environment.",
            metadata={"exception_type": type(exc).__name__},
        )

    @staticmethod
    def _validation_result(exc: ValidationError) -> ToolResult:
        """把 Pydantic 错误压缩成不含输入值/文档 URL 的稳定字段列表。"""
        return ToolResult(
            success=False,
            content="Tool argument validation failed.",
            error="invalid_tool_arguments",
            suggestion="Check the tool schema and retry with valid arguments.",
            metadata={
                "validation_errors": [
                    {
                        "location": ".".join(str(part) for part in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                    for error in exc.errors(include_url=False, include_input=False)
                ]
            },
        )

    async def _run_observers(
        self,
        call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> dict[str, dict[str, Any]]:
        """并发、独立限时运行全部 Observer，失败只写 post_actions 诊断。

        Observer 只在原工具成功后运行；其超时或异常不会倒置工具成功事实。
        """

        async def run(
            observer: ToolResultObserver,
        ) -> tuple[str, dict[str, Any] | None]:
            """限时执行单个 Observer，并把所有非取消失败转成诊断。"""
            details: dict[str, Any] | None
            deadline = asyncio.timeout(context.tool_timeout_seconds)
            try:
                async with deadline:
                    details = await observer.after_result(call, result, context)
            except TimeoutError as exc:
                if deadline.expired():
                    details = {
                        "status": "failed",
                        "error": "observer_timeout",
                        "timeout_seconds": context.tool_timeout_seconds,
                    }
                else:
                    details = {
                        "status": "failed",
                        "error": "observer_error",
                        "exception_type": type(exc).__name__,
                    }
            except Exception as exc:
                details = {
                    "status": "failed",
                    "error": "observer_error",
                    "exception_type": type(exc).__name__,
                }
            return observer.name, details

        completed = await asyncio.gather(
            *(run(observer) for observer in self.observers)
        )
        return {name: details for name, details in completed if details is not None}

    @staticmethod
    def _sandbox_policy(context: TurnContext) -> SandboxPolicy:
        """从 TurnContext 快照构造本次 effect 检查策略。"""
        return SandboxPolicy(
            mode=context.sandbox_mode,
            network_access=context.network_access,
        )

    @staticmethod
    def _limit_output(
        result: ToolResult,
        max_chars: int,
        max_tokens: int,
    ) -> ToolResult:
        """按字符、Token、结构化字段和最终 model_content 四层限制输出。

        content 先限字符再限 Token；suggestion 使用最多四分之一预算；data 与
        metadata 超大时替换为截断摘要；最后对真正发给模型的序列化文本二分
        搜索最大可保留 content，避免包装字段使总量重新超限。
        """
        content = result.content
        metadata = dict(result.metadata)
        data = result.data
        suggestion = result.suggestion
        original_tokens = estimate_text_tokens(content)

        if len(content) > max_chars:
            content = content[:max_chars]
            metadata.update(
                {
                    "content_truncated": True,
                    "original_content_chars": len(result.content),
                }
            )

        if estimate_text_tokens(content) > max_tokens:
            content = truncate_text_to_tokens(content, max_tokens)
            metadata.update(
                {
                    "content_truncated": True,
                    "original_content_chars": len(result.content),
                    "original_content_tokens": original_tokens,
                }
            )

        if suggestion:
            suggestion_tokens = max(1, min(64, max_tokens // 4))
            limited_suggestion = truncate_text_to_tokens(
                suggestion,
                suggestion_tokens,
            )
            if limited_suggestion != suggestion:
                suggestion = limited_suggestion
                metadata["suggestion_truncated"] = True

        if data is not None:
            data = ToolRunner._limit_mapping(data, max_chars, label="data")

        metadata_chars = ToolRunner._json_chars(metadata)
        if metadata_chars > max_chars:
            preserved = {
                key: metadata[key]
                for key in (
                    "content_truncated",
                    "original_content_chars",
                    "outcome_unknown",
                    "retry_safe",
                )
                if key in metadata
            }
            metadata = {
                **preserved,
                "metadata_truncated": True,
                "original_metadata_chars": metadata_chars,
            }

        limited = result.model_copy(
            update={
                "content": content,
                "data": data,
                "metadata": metadata,
                "suggestion": suggestion,
            }
        )
        return ToolRunner._fit_model_content(limited, max_tokens=max_tokens)

    @staticmethod
    def _fit_model_content(result: ToolResult, *, max_tokens: int) -> ToolResult:
        """二分查找能与结构化包装共同装入 Token 上限的最长 content 前缀。"""
        if estimate_text_tokens(result.model_content()) <= max_tokens:
            return result

        if result.metadata.get("content_truncated") is not True:
            result = result.model_copy(
                update={
                    "metadata": {
                        **result.metadata,
                        "content_truncated": True,
                        "original_content_chars": len(result.content),
                        "original_content_tokens": estimate_text_tokens(result.content),
                    }
                }
            )

        candidates = [result]
        if result.suggestion is not None:
            candidates.append(result.model_copy(update={"suggestion": None}))

        for candidate in candidates:
            empty = candidate.model_copy(update={"content": ""})
            if estimate_text_tokens(empty.model_content()) > max_tokens:
                continue
            low = 0
            high = len(candidate.content)
            best = empty
            while low <= high:
                middle = (low + high) // 2
                attempted = candidate.model_copy(
                    update={"content": candidate.content[:middle]}
                )
                if estimate_text_tokens(attempted.model_content()) <= max_tokens:
                    best = attempted
                    low = middle + 1
                else:
                    high = middle - 1
            return best

        return result.model_copy(update={"content": "", "suggestion": None})

    @staticmethod
    def _limit_mapping(
        value: dict[str, Any], max_chars: int, *, label: str
    ) -> dict[str, Any]:
        """JSON 字符数超限时用 label 对应的最小截断摘要替换 mapping。"""
        original_chars = ToolRunner._json_chars(value)
        if original_chars <= max_chars:
            return value
        return {
            f"{label}_truncated": True,
            f"original_{label}_chars": original_chars,
        }

    @staticmethod
    def _json_chars(value: object) -> int:
        """按实际紧凑 Unicode JSON 序列化估算结构化值字符数。"""
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )

    @staticmethod
    def _finished_payload(
        call: ToolCall,
        result: ToolResult,
        *,
        started_at: float,
        approval_wait_ms: int = 0,
        execution_ms: int = 0,
        observer_ms: int = 0,
    ) -> dict[str, Any]:
        """构造最终事件，并把总耗时拆为治理/审批/执行/Observer。

        governance 用总耗时减去其他已测阶段且下限为零，包含 Registry、参数、
        effect 检查、事件生成和输出治理等框架成本。
        """
        total_ms = int((monotonic() - started_at) * 1000)
        governance_ms = max(
            0,
            total_ms - approval_wait_ms - execution_ms - observer_ms,
        )
        return {
            "call_id": call.call_id,
            "name": call.name,
            "result": result.model_dump(mode="json"),
            "duration_ms": total_ms,
            "timings_ms": {
                "governance": governance_ms,
                "approval_wait": approval_wait_ms,
                "execution": execution_ms,
                "observers": observer_ms,
                "total": total_ms,
            },
        }
