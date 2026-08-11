from __future__ import annotations

import asyncio

from codecraft.cli.ui.event_renderer import RuntimeEventRenderer
from codecraft.core.ids import new_id
from codecraft.core.thread import AgentThread
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.input import SessionInput


async def submit_user_message(
    thread: AgentThread,
    renderer: RuntimeEventRenderer,
    text: str,
) -> int:
    """提交一条新用户消息并消费其 Turn，返回 shell exit code。"""
    await thread.submit(SessionInput.user_message(new_id("inp_"), text))
    return await consume_turn(thread, renderer)


async def consume_turn(thread: AgentThread, renderer: RuntimeEventRenderer) -> int:
    """消费直到 Turn 终态；审批事件同步询问并作为旁路输入回送。

    成功返回 0，中止返回 1；KeyboardInterrupt 会先拒绝 pending approvals、
    中止并关闭 Thread，再重新抛出给 Typer/终端处理。
    """
    try:
        while True:
            event = await thread.next_event()
            if event.type == RuntimeEventType.APPROVAL_REQUESTED:
                decision = await renderer.request_approval(event)
                await thread.submit(decision)
                continue

            await renderer.render(event)

            if event.type == RuntimeEventType.TURN_FINISHED:
                renderer.ensure_newline()
                await thread.wait_until_idle()
                return 0
            if event.type == RuntimeEventType.TURN_ABORTED:
                renderer.ensure_newline()
                await thread.wait_until_idle()
                return 1
    except KeyboardInterrupt:
        await shutdown_thread(thread)
        raise


async def shutdown_thread(thread: AgentThread) -> None:
    """拒绝所有待审批项、中止/关闭 Session，并短暂等待后台 Task 收口。"""
    for approval in thread.list_pending_approvals():
        await thread.submit(
            SessionInput.approval_decision(
                new_id("inp_"),
                approval_id=approval.approval_id,
                approved=False,
                reason="interrupted by CLI",
            )
        )
    await thread.interrupt("interrupted by CLI")
    await thread.close()
    try:
        await asyncio.wait_for(thread.wait_until_idle(), timeout=1)
    except TimeoutError:
        return
