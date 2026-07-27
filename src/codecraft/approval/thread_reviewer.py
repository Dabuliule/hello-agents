from __future__ import annotations

import asyncio

from codecraft.approval.manager import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalReviewer,
)


class ThreadApprovalReviewer(ApprovalReviewer):
    """把 approval request 挂起，等待同一个 thread 提交决定。

    review() 创建 Future 并暂停 tool 执行；decide() 由用户输入触发，把结果
    填回 Future，让 ToolRunner 继续往下走。
    """

    def __init__(self) -> None:
        self.pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self.requests: dict[str, ApprovalRequest] = {}

    async def review(self, request: ApprovalRequest) -> ApprovalDecision:
        """登记审批请求，并等待用户决定。"""
        if request.approval_id in self.pending:
            raise RuntimeError(f"duplicate approval request: {request.approval_id}")

        future: asyncio.Future[ApprovalDecision] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending[request.approval_id] = future
        self.requests[request.approval_id] = request
        try:
            return await future
        finally:
            self.pending.pop(request.approval_id, None)
            self.requests.pop(request.approval_id, None)

    def decide(self, decision: ApprovalDecision) -> None:
        """用用户决定唤醒对应的 approval Future。"""
        future = self.pending.get(decision.approval_id)
        if future is None:
            raise KeyError(f"unknown pending approval: {decision.approval_id}")
        if future.done():
            raise RuntimeError(f"approval already decided: {decision.approval_id}")
        future.set_result(decision)

    def list_pending(self) -> list[ApprovalRequest]:
        return [
            self.requests[approval_id]
            for approval_id in self.pending
            if approval_id in self.requests
        ]
