from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from codecraft.approval.manager import ApprovalDecision, ApprovalManager
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.core.errors import CodecraftError
from codecraft.core.conversation import Conversation
from codecraft.core.event_bus import EventBus
from codecraft.core.ids import new_id
from codecraft.core.session_store import SessionStore
from codecraft.core.turn import Turn
from codecraft.llm.base import LLMProvider
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import (
    ApprovalDecisionPayload,
    InterruptPayload,
    SessionInput,
    SessionInputType,
)
from codecraft.schema.session import SessionConfig
from codecraft.skill import SkillRegistry
from codecraft.tool.registry import ToolRegistry
from codecraft.tool.observer import ToolResultObserver
from codecraft.tool.runner import ToolRunner


class SessionStatus(StrEnum):
    """Session 是否可调度、正执行、处理中断或永久关闭。"""

    IDLE = "idle"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    CLOSED = "closed"


class Session:
    """一个可持续追加事件的 agent 会话。

    ``Session`` 是运行时的长生命周期状态和调度中心：接收输入、串行启动 Turn、
    分发审批决定，并把所有关键状态写成 RuntimeEvent。它拥有 Conversation、
    active Turn task、状态机和事件序号；外层 UI 只通过 AgentThread 输入与事件流
    交互，不直接修改这些内部状态。
    """

    def __init__(
        self,
        *,
        config: SessionConfig,
        session_store: SessionStore,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        approval_manager: ApprovalManager | None = None,
        tool_result_observers: Sequence[ToolResultObserver] | None = None,
        event_bus: EventBus | None = None,
        conversation: Conversation | None = None,
        seq: int = 0,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """装配会话状态、依赖、三类锁、输入队列和可选恢复历史。

        Args:
            config: 当前 Session 固定使用并可持久化的执行配置。
            session_store: 所有 RuntimeEvent 的 JSONL 持久化边界。
            llm_provider: 已由 Runtime 按配置选择的模型 Provider。
            tool_registry: 已完成启动和动态工具发现的 Registry。
            approval_manager: 可选的共享审批管理器。
            tool_result_observers: ToolRunner 成功结果的后处理器。
            event_bus: 可选外部总线；缺失时为该 Session 创建独立 EventBus。
            conversation: Resume 时重建的历史；新 Session 使用空 Conversation。
            seq: 已持久化的最后事件序号；新 Session 从零开始。
            skill_registry: 当前 Runtime 发现的 Skill 集合。

        ``seq`` 应等于恢复日志最后事件序号；默认 Reviewer 是能接收 Thread
        旁路审批的 ThreadApprovalReviewer。Session 始终只保存一个 active_turn。
        构造阶段不启动 Turn，也不发事件，确保 AgentThread 可以先完成订阅。
        """
        self.session_id = config.session_id
        self.config = config
        self.conversation = conversation or Conversation()
        self.input_queue: asyncio.Queue[SessionInput] = asyncio.Queue()
        self.active_turn: Turn | None = None
        self.status = SessionStatus.IDLE
        self.event_bus = event_bus or EventBus()
        self.session_store = session_store
        self.seq = seq
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.skill_registry = (
            skill_registry if skill_registry is not None else SkillRegistry()
        )
        self.approval_manager = approval_manager or ApprovalManager(
            reviewer=ThreadApprovalReviewer()
        )
        self.tool_runner = ToolRunner(
            tool_registry,
            approval_manager=self.approval_manager,
            observers=tool_result_observers,
        )
        self._emit_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._closed_event_emitted = False

    async def submit(self, input: SessionInput) -> str:
        """提交用户输入、审批结果或中断请求。"""
        if input.type == SessionInputType.USER_MESSAGE:
            async with self._state_lock:
                if self.status == SessionStatus.CLOSED:
                    raise RuntimeError("session is closed")
                self.input_queue.put_nowait(input)
            await self.start_turn_if_idle()
            return input.input_id

        if input.type == SessionInputType.INTERRUPT:
            if not isinstance(input.payload, InterruptPayload):
                raise TypeError("interrupt input has the wrong payload type")
            await self.interrupt(input.payload.reason)
            return input.input_id

        if input.type == SessionInputType.APPROVAL_DECISION:
            if self.status == SessionStatus.CLOSED:
                raise RuntimeError("session is closed")
            self.submit_approval_decision(input)
            return input.input_id

        raise ValueError(f"unsupported session input type: {input.type}")

    def submit_approval_decision(self, input: SessionInput) -> None:
        """把用户审批结果交给正在等待的 reviewer。"""
        if not isinstance(input.payload, ApprovalDecisionPayload):
            raise TypeError("approval input has the wrong payload type")
        decision = ApprovalDecision(
            approval_id=input.payload.approval_id,
            approved=input.payload.approved,
            reviewer="user",
            reason=input.payload.reason,
        )
        reviewer = self.approval_manager.reviewer
        if not isinstance(reviewer, ThreadApprovalReviewer):
            raise RuntimeError(
                "current approval reviewer does not accept thread decisions"
            )
        reviewer.decide(decision)

    async def start_turn_if_idle(self) -> None:
        """如果当前空闲，就从输入队列取一条消息启动新 turn。"""
        async with self._state_lock:
            if self.status != SessionStatus.IDLE or self._runner_task is not None:
                return
            if self.input_queue.empty():
                return

            user_input = self.input_queue.get_nowait()
            turn = Turn(
                session=self,
                turn_id=new_id("turn_"),
            )
            self.active_turn = turn
            self.status = SessionStatus.RUNNING
            self._runner_task = asyncio.create_task(
                self._run_turn(turn, user_input),
                name=f"codecraft-turn-{turn.turn_id}",
            )

    async def emit(
        self,
        event_type: RuntimeEventType,
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> RuntimeEvent:
        """持久化并广播一个 RuntimeEvent。

        seq 是 session 日志的顺序号；写入失败时回滚 seq，避免后续事件出现
        不连续的编号。
        """
        async with self._emit_lock:
            self.seq += 1
            event = RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=self.session_id,
                turn_id=turn_id,
                seq=self.seq,
                type=event_type,
                payload=payload or {},
            )
            try:
                await self.session_store.append_event(event)
            except asyncio.CancelledError:
                # SessionStore completes a scheduled append before surfacing
                # cancellation, so finish the matching broadcast as well.
                await self.event_bus.emit(event)
                raise
            except Exception:
                self.seq -= 1
                raise
            await self.event_bus.emit(event)
            return event

    async def interrupt(self, reason: str) -> None:
        """取消当前 turn，并等待后台 task 完成清理。"""
        async with self._state_lock:
            if self.status == SessionStatus.CLOSED:
                return
            task = self._runner_task
            if task is None:
                return
            self.status = SessionStatus.INTERRUPTED
            if task.cancelling() == 0:
                task.cancel(reason)
        await asyncio.shield(task)

    async def close(self) -> None:
        """串行、幂等关闭；先取消并等待 active Turn，再持久化 SESSION_CLOSED。"""
        async with self._close_lock:
            if self._closed_event_emitted:
                return
            async with self._state_lock:
                self.status = SessionStatus.CLOSED
                task = self._runner_task
                if task is not None and task.cancelling() == 0:
                    task.cancel("session_closed")
            if task is not None:
                await asyncio.shield(task)
            await self.emit(RuntimeEventType.SESSION_CLOSED)
            self._closed_event_emitted = True

    async def wait_until_idle(self) -> None:
        """等待当前及已排队 turn 全部处理完成。"""
        while True:
            async with self._state_lock:
                task = self._runner_task
            if task is None:
                return
            await asyncio.shield(task)

    async def _run_turn(self, turn: Turn, user_input: SessionInput) -> None:
        """包装 turn.run，确保异常也会变成可追踪的事件。"""
        deadline = asyncio.timeout(turn.context.turn_timeout_seconds)
        try:
            async with deadline:
                await turn.run(user_input)
        except TimeoutError as exc:
            if deadline.expired():
                await turn.abort(
                    "turn_timeout",
                    "Turn exceeded the configured execution deadline.",
                    metadata={
                        "timeout_seconds": turn.context.turn_timeout_seconds,
                    },
                )
            else:
                await self._abort_from_exception(turn, exc)
        except asyncio.CancelledError as exc:
            reason = str(exc.args[0]) if exc.args else "turn_cancelled"
            await turn.abort(reason, reason)
        except Exception as exc:
            await self._abort_from_exception(turn, exc)
        finally:
            current_task = asyncio.current_task()
            async with self._state_lock:
                if self._runner_task is current_task:
                    self._runner_task = None
                if self.active_turn is turn:
                    self.active_turn = None
                should_continue = self.status != SessionStatus.CLOSED
                if should_continue:
                    self.status = SessionStatus.IDLE
            if should_continue:
                await self.start_turn_if_idle()

    async def _abort_from_exception(self, turn: Turn, exc: Exception) -> None:
        """先记录 ERROR 事件，再用同一稳定信息收口 TURN_ABORTED。"""
        error_payload = self._error_payload(exc)
        await self.emit(
            RuntimeEventType.ERROR,
            error_payload,
            turn_id=turn.turn_id,
        )
        await turn.abort(
            error_payload["code"],
            error_payload["message"],
            metadata=error_payload.get("metadata", {}),
        )

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, Any]:
        """保留 CodecraftError 机器字段，未知异常归一为 runtime_error。"""
        if isinstance(exc, CodecraftError):
            return {
                "code": exc.code,
                "message": exc.message,
                "metadata": exc.metadata,
                "suggestion": exc.suggestion,
            }
        return {
            "code": "runtime_error",
            "message": str(exc),
            "metadata": {},
            "suggestion": None,
        }
