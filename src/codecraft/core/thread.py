"""面向 CLI/TUI 的 Session facade 与 RuntimeEvent 异步队列。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from codecraft.core.session import Session
from codecraft.approval.manager import ApprovalRequest
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionSnapshot


class AgentThread:
    """面向 CLI/UI 的 Session facade，不引入第二个持久化身份。

    ``AgentThread`` 把 Session 的事件总线转成 FIFO 异步队列，让调用方可以像
    消费 stream 一样读取 RuntimeEvent，同时隐藏 Session 的状态锁、Turn task 和
    Conversation mutation。Thread 是进程内句柄；可恢复身份始终是 ``session_id``。
    """

    def __init__(self, session: Session) -> None:
        """订阅 Session EventBus，并把后续事件捕获进 FIFO 异步队列。

        Args:
            session: 已构造但尚未发出首个公开事件的 Session。

        Runtime 必须在 ``SESSION_STARTED``/``SESSION_RESTORED`` 之前创建 Thread，
        否则队列会漏掉会话首事件。Queue 保留 EventBus 的发布顺序，并为 UI 消费
        速度与 Session 事件产生速度提供异步解耦。
        """
        self.session = session
        self._events: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self.session.event_bus.subscribe(self._capture_event)

    async def submit(self, input: SessionInput) -> str:
        """提交用户消息、中止或审批输入，并返回原 ``input_id``。

        Thread 不解释输入内容；类型分发、状态检查和 Turn 调度全部委托给 Session。
        返回 input ID 便于调用方关联请求，Turn ID 则由 Session 启动 Turn 时另行生成。
        """
        return await self.session.submit(input)

    async def next_event(self) -> RuntimeEvent:
        """等待并取出下一条 RuntimeEvent；没有事件时挂起而不轮询。"""
        return await self._events.get()

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        """按发布顺序持续产出事件，收到 ``SESSION_CLOSED`` 后结束迭代。"""
        while True:
            event = await self.next_event()
            yield event
            if event.type == RuntimeEventType.SESSION_CLOSED:
                return

    async def interrupt(self, reason: str = "user_interrupt") -> None:
        """请求幂等中止当前 active Turn，并等待其终态清理完成。

        Args:
            reason: 写入 ``TURN_ABORTED`` 的稳定取消原因。

        中止只结束当前 Turn；Session 未关闭时仍可处理已排队或后续用户消息。
        """
        await self.session.interrupt(reason)

    async def close(self) -> None:
        """永久关闭 Session，并在必要时先中止和收口 active Turn。

        返回前 ``SESSION_CLOSED`` 已持久化；关闭后不能再提交用户消息或审批决定。
        Runtime 拥有的 Provider/Tool 资源仍由 ``AgentRuntime.close`` 单独释放。
        """
        await self.session.close()

    def list_pending_approvals(self) -> list[ApprovalRequest]:
        """返回当前等待用户处理的审批请求。"""
        reviewer = self.session.approval_manager.reviewer
        if isinstance(reviewer, ThreadApprovalReviewer):
            return reviewer.list_pending()
        return []

    async def read_snapshot(self) -> SessionSnapshot:
        """从 Store 读取当前已持久化事件，返回一致 Snapshot。"""
        events = await self.session.session_store.load_events(self.session.session_id)
        return SessionSnapshot(config=self.session.config, events=events)

    async def wait_until_idle(self) -> None:
        """等待当前后台 turn 结束。

        测试和命令行一次性执行会用它确保事件都写完后再退出。
        """
        await self.session.wait_until_idle()

    async def _capture_event(self, event: RuntimeEvent) -> None:
        """EventBus handler：按发布顺序将事件放入 Thread 队列。"""
        await self._events.put(event)
