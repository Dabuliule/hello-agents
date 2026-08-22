from __future__ import annotations

import asyncio

from codecraft.approval.manager import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalReviewer,
)


class ThreadApprovalReviewer(ApprovalReviewer):
    """用 approval ID → Future 映射挂起工具，等待 Thread 控制面提交决定。

    ``prepare()`` 在请求事件发布前创建 Future；``review()`` 暂停原工具协程；
    同步 ``decide()`` 由 Session 的旁路控制输入触发，把结果填回 Future，让同一
    协程从 await 点继续。映射操作间没有 await，依靠单一 asyncio event loop 的
    协作式调度保持一致，不需要 Lock。
    """

    def __init__(self) -> None:
        """创建 approval ID 到 Future、请求快照的两个同步映射。"""
        self.pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self.requests: dict[str, ApprovalRequest] = {}

    def prepare(self, request: ApprovalRequest) -> None:
        """同步登记 Future，使事件一旦可见就能立即接收用户决定。

        Raises:
            RuntimeError: approval ID 已处于 pending，拒绝覆盖原等待者。

        本方法没有 await；Future 和请求快照会在同一个 event-loop 调度片段中一起
        可见，从而建立 ``prepare → publish → decide`` 的 happens-before 顺序。
        """
        if request.approval_id in self.pending:
            raise RuntimeError(f"duplicate approval request: {request.approval_id}")

        future: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending[request.approval_id] = future
        self.requests[request.approval_id] = request

    def cancel(self, request: ApprovalRequest) -> None:
        """幂等移除同一请求的 Future，并取消尚未完成的等待。"""
        current_request = self.requests.get(request.approval_id)
        if current_request != request:
            return
        future = self.pending.pop(request.approval_id, None)
        self.requests.pop(request.approval_id, None)
        if future is not None and not future.done():
            future.cancel()

    async def review(self, request: ApprovalRequest) -> ApprovalDecision:
        """等待 prepare 创建的 Future，通过 finally 覆盖所有退出路径的清理。

        Returns:
            ``decide`` 为同 approval ID Future 填入的允许或拒绝决定。

        Raises:
            asyncio.CancelledError: Turn interrupt/close 取消原工具协程。

        正常决定、Reviewer 外层 timeout 和 Turn 取消都会执行 finally，防止 UI
        永久看到一条已经没有等待协程的 pending approval。
        """
        future = self.pending.get(request.approval_id)
        if future is None or self.requests.get(request.approval_id) != request:
            raise RuntimeError(
                f"approval request was not prepared: {request.approval_id}"
            )
        try:
            return await future
        finally:
            self.cancel(request)

    def decide(self, decision: ApprovalDecision) -> None:
        """同步填充对应 Future，使挂起的 ``review`` 在下一次调度时恢复。

        Raises:
            KeyError: approval ID 未登记或已经在 finally 中清理。
            RuntimeError: 同一 Future 已有决定，拒绝二次审批。

        方法刻意不 await，Session 可以在不进入 USER_MESSAGE 队列的控制面路径中
        原子检查并 set_result。
        """
        future = self.pending.get(decision.approval_id)
        if future is None:
            raise KeyError(f"unknown pending approval: {decision.approval_id}")
        if future.done():
            raise RuntimeError(f"approval already decided: {decision.approval_id}")
        future.set_result(decision)

    def list_pending(self) -> list[ApprovalRequest]:
        """按 Future 登记顺序返回仍等待用户决定的请求快照。"""
        return [
            self.requests[approval_id]
            for approval_id in self.pending
            if approval_id in self.requests
        ]
