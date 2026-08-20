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
        """按数据面或控制面语义分发一条结构化输入。

        Args:
            input: 已通过 payload/type 一致性校验的 SessionInput。

        Returns:
            原样返回 ``input.input_id``，供调用方关联本次提交。

        Raises:
            RuntimeError: 已关闭 Session 收到用户消息或审批决定。
            TypeError: input type 与具体 payload 模型不一致。
            ValueError: 收到未知输入类型。

        USER_MESSAGE 在 ``_state_lock`` 内进入 FIFO 队列，释放锁后尝试启动 Turn；
        INTERRUPT 和 APPROVAL_DECISION 是当前 Turn 的旁路控制输入，不进入队列。
        如果审批决定排在用户消息队列中，正在等待该决定的 Turn 将无法结束，形成
        自我死锁。
        """
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
        """把用户审批结果直接交给正在等待同一 approval ID 的 Reviewer。

        Args:
            input: payload 必须是 ``ApprovalDecisionPayload`` 的控制输入。

        Raises:
            TypeError: payload 类型不匹配。
            RuntimeError: 当前 Reviewer 不支持由 Thread 旁路提交决定。

        本方法不修改输入队列或创建 Turn；Reviewer 内部 Future 被唤醒后，原 active
        Turn 才能从工具审批等待点继续执行。
        """
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
        """原子取得唯一执行权，从 FIFO 队列取一条消息启动后台 Turn。

        ``_state_lock`` 把检查 IDLE、检查 ``_runner_task``、出队、更新状态和发布
        Task 变成一个不可交错的临界区。多个并发 submit 即使都调用本方法，也只有
        第一个持锁者能创建 Turn；其余调用看到 RUNNING 或已有 Task 后直接返回。

        Turn 通过 ``asyncio.create_task`` 在后台运行，因此 submit 不等待模型完成。
        当前 Task 在 ``_run_turn`` 的 finally 中清理后，会再次调用本方法继续处理
        队列中的下一条用户消息。
        """
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
        """幂等取消当前 Turn，并 shield 等待其终态事件和状态清理完成。

        中断是旁路控制操作，不进入用户消息队列。锁内只改变状态并发出 cancel，
        锁外等待 Task，避免 Turn finally 获取 ``_state_lock`` 时发生死锁。
        """
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
        """串行、幂等关闭；先收口 active Turn，再持久化 SESSION_CLOSED。

        ``_close_lock`` 保证并发 close 只有一个协程执行关闭协议。锁内先把 status
        设为 CLOSED，再取得并取消当前 Task；这样 Task finally 不会启动队列中的
        下一条消息。等待发生在 ``_state_lock`` 外，且使用 shield 保护 Turn 的
        ``TURN_ABORTED`` 记录和 finally 清理不被等待者取消传播打断。

        ``SESSION_CLOSED`` 只在 active Task 完全结束后发出，因此事件日志中的
        生命周期顺序稳定为 Turn 终态在前、Session 终态在后。重复调用通过
        ``_closed_event_emitted`` 直接返回，不会产生多个关闭事件。
        """
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
        """等待当前及已排队 Turn 全部处理完成。

        每轮在锁内读取当前 Task、锁外 shield 等待。Task finally 可能立即启动下一
        条排队消息，因此方法循环检查，直到 ``_runner_task`` 真正变为 ``None``。
        ``shield`` 防止等待者自身被取消时顺带取消 Session 正在执行的 Turn。
        """
        while True:
            async with self._state_lock:
                task = self._runner_task
            if task is None:
                return
            await asyncio.shield(task)

    async def _run_turn(self, turn: Turn, user_input: SessionInput) -> None:
        """在 Session 所有的后台 Task 中限时运行 Turn，并统一收口生命周期。

        Turn 超时、显式取消和普通异常分别转换为稳定的终态事件。``finally`` 在
        ``_state_lock`` 内仅清理仍属于当前 Task/Turn 的引用，防止过期 Task 覆盖
        新状态；Session 未关闭时恢复 IDLE，并在锁外调度下一条排队消息。

        Session 在这一层拥有整轮 deadline，所以模型、工具、Observer 和审批等待
        都不能逃出 ``turn_timeout_seconds``。
        """
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
