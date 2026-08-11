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
    """面向 CLI/UI 的 session facade。

    `AgentThread` 把 Session 的事件总线转成一个异步队列，让调用方可以像读
    stream 一样消费事件，同时隐藏 session 的调度细节。
    """

    def __init__(self, session: Session) -> None:
        """订阅 Session EventBus，并把每个事件捕获进 FIFO 异步队列。"""
        self.session = session
        self._events: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self.session.event_bus.subscribe(self._capture_event)

    async def submit(self, input: SessionInput) -> str:
        """提交消息/中止/审批输入，返回新 Turn ID 或受控操作 ID。"""
        return await self.session.submit(input)

    async def next_event(self) -> RuntimeEvent:
        """等待并取出下一条 RuntimeEvent。"""
        return await self._events.get()

    async def events(self) -> AsyncIterator[RuntimeEvent]:
        """持续产出事件，直到收到 SESSION_CLOSED。"""
        while True:
            event = await self.next_event()
            yield event
            if event.type == RuntimeEventType.SESSION_CLOSED:
                return

    async def interrupt(self, reason: str = "user_interrupt") -> None:
        """请求幂等中止当前 active Turn。"""
        await self.session.interrupt(reason)

    async def close(self) -> None:
        """关闭 Session，并在必要时先中止 active Turn。"""
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
