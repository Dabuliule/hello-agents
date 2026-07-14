from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
import json
from time import monotonic

from pydantic import ValidationError

from codecraft.approval.manager import ApprovalManager
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
    type: RuntimeEventType
    payload: dict


class ToolRunner:
    """统一执行 tool call，并把执行过程转成 RuntimeEvent。

    调用顺序是：参数 schema 校验、sandbox effect 检查、approval 检查、真正
    执行 tool。每一步失败都会变成 ToolResult，而不是让异常直接穿透到 turn。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        approval_manager: ApprovalManager | None = None,
        observers: Sequence[ToolResultObserver] | None = None,
    ) -> None:
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
        """运行一个 tool call，并按执行阶段产出事件。"""
        yield ToolRunnerEvent(
            RuntimeEventType.TOOL_CALL_STARTED,
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            },
        )

        started_at = monotonic()
        result: ToolResult
        approved = False
        approval_wait_ms = 0
        execution_ms = 0
        observer_ms = 0
        execution_deadline: asyncio.Timeout | None = None
        try:
            tool = self.registry.get(call.name)
            args = tool.args_schema.model_validate(call.arguments)
            sandbox_evaluation = self._sandbox_policy(context).evaluate_effects(
                tool.effects
            )
            if not sandbox_evaluation.allowed:
                # sandbox 是硬边界；不进入 approval，也不执行 tool。
                result = ToolResult(
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
                result = self._limit_output(
                    result,
                    context.max_tool_output_chars,
                    context.max_tool_output_tokens,
                )
                yield ToolRunnerEvent(
                    RuntimeEventType.TOOL_CALL_FINISHED,
                    self._finished_payload(
                        call,
                        result,
                        started_at=started_at,
                    ),
                )
                return

            evaluation = await self.approval_manager.evaluate(tool, call, args, context)
            if evaluation.requires_approval:
                # approval 是可交互边界，通常由 UI 把请求展示给用户处理。
                approval_request = self.approval_manager.build_request(
                    call=call,
                    context=context,
                    evaluation=evaluation,
                )
                yield ToolRunnerEvent(
                    RuntimeEventType.APPROVAL_REQUESTED,
                    approval_request.model_dump(mode="json"),
                )
                approval_started_at = monotonic()
                approval_error: str | None = None
                approval_exception_type: str | None = None
                approval_deadline = asyncio.timeout(context.approval_timeout_seconds)
                try:
                    async with approval_deadline:
                        approval_decision = await self.approval_manager.request(
                            approval_request
                        )
                except TimeoutError as exc:
                    approval_error = (
                        "approval_timeout"
                        if approval_deadline.expired()
                        else "approval_error"
                    )
                    approval_exception_type = type(exc).__name__
                    approval_decision = (
                        self.approval_manager.build_reviewer_failure_decision(
                            approval_request,
                            timed_out=approval_deadline.expired(),
                        )
                    )
                except Exception as exc:
                    approval_error = "approval_error"
                    approval_exception_type = type(exc).__name__
                    approval_decision = (
                        self.approval_manager.build_reviewer_failure_decision(
                            approval_request,
                            timed_out=False,
                        )
                    )
                finally:
                    approval_wait_ms = int((monotonic() - approval_started_at) * 1000)
                yield ToolRunnerEvent(
                    RuntimeEventType.APPROVAL_DECIDED,
                    approval_decision.model_dump(mode="json"),
                )
                if not approval_decision.approved:
                    result = ToolResult(
                        success=False,
                        content=(
                            "Tool approval timed out."
                            if approval_error == "approval_timeout"
                            else "Tool execution denied by approval."
                        ),
                        error=approval_error or "approval_denied",
                        suggestion=approval_decision.reason,
                        metadata={
                            "approval_id": approval_decision.approval_id,
                            "tool": call.name,
                            **(
                                {"exception_type": approval_exception_type}
                                if approval_exception_type is not None
                                else {}
                            ),
                        },
                    )
                    result = self._limit_output(
                        result,
                        context.max_tool_output_chars,
                        context.max_tool_output_tokens,
                    )
                    yield ToolRunnerEvent(
                        RuntimeEventType.TOOL_CALL_FINISHED,
                        self._finished_payload(
                            call,
                            result,
                            started_at=started_at,
                            approval_wait_ms=approval_wait_ms,
                        ),
                    )
                    return
                approved = True
            execution_started_at = monotonic()
            execution_deadline = asyncio.timeout(context.tool_timeout_seconds)
            try:
                async with execution_deadline:
                    result = await tool.arun(
                        args,
                        ToolContext(
                            context=context,
                            call=call,
                            approved=approved,
                            command_decision=evaluation.command_decision,
                        ),
                    )
            finally:
                execution_ms = int((monotonic() - execution_started_at) * 1000)
        except TimeoutError as exc:
            if execution_deadline is not None and execution_deadline.expired():
                result = ToolResult(
                    success=False,
                    content="Tool execution timed out.",
                    error="tool_timeout",
                    suggestion="Retry with a narrower operation or increase the tool timeout.",
                    metadata={"timeout_seconds": context.tool_timeout_seconds},
                )
            else:
                result = ToolResult(
                    success=False,
                    content="Tool execution failed.",
                    error="tool_execution_error",
                    suggestion="Check the tool arguments or runtime environment.",
                    metadata={"exception_type": type(exc).__name__},
                )
        except ValidationError as exc:
            result = ToolResult(
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
        except CodecraftError as exc:
            result = ToolResult(
                success=False,
                content=exc.message,
                error=exc.code,
                suggestion=exc.suggestion,
                metadata=exc.metadata,
            )
        except Exception as exc:
            result = ToolResult(
                success=False,
                content="Tool execution failed.",
                error="tool_execution_error",
                suggestion="Check the tool arguments, workspace permissions, or runtime environment.",
                metadata={"exception_type": type(exc).__name__},
            )

        if result.success and self.observers:
            observer_started_at = monotonic()
            post_actions = await self._run_observers(call, result, context)
            observer_ms = int((monotonic() - observer_started_at) * 1000)
            if post_actions:
                result.metadata["post_actions"] = post_actions

        result = self._limit_output(
            result,
            context.max_tool_output_chars,
            context.max_tool_output_tokens,
        )
        yield ToolRunnerEvent(
            RuntimeEventType.TOOL_CALL_FINISHED,
            self._finished_payload(
                call,
                result,
                started_at=started_at,
                approval_wait_ms=approval_wait_ms,
                execution_ms=execution_ms,
                observer_ms=observer_ms,
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

    async def _run_observers(
        self,
        call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> dict:
        async def run(observer: ToolResultObserver) -> tuple[str, dict | None]:
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
        return SandboxPolicy(
            mode=context.sandbox_mode,
            workspace_roots=context.workspace_roots,
            network_access=context.network_access,
        )

    @staticmethod
    def _limit_output(
        result: ToolResult,
        max_chars: int,
        max_tokens: int,
    ) -> ToolResult:
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
                for key in ("content_truncated", "original_content_chars")
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
    def _limit_mapping(value: dict, max_chars: int, *, label: str) -> dict:
        original_chars = ToolRunner._json_chars(value)
        if original_chars <= max_chars:
            return value
        return {
            f"{label}_truncated": True,
            f"original_{label}_chars": original_chars,
        }

    @staticmethod
    def _json_chars(value: object) -> int:
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
    ) -> dict:
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
