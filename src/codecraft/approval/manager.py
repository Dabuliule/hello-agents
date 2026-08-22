from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel

from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.ids import new_id
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox.command_policy import CommandDecision, CommandPolicy, CommandRisk
from codecraft.schema.tool import ToolCall, ToolEffect
from codecraft.tool.base import BaseTool


class ApprovalEvaluation(BaseModel):
    """工具治理阶段的审批需求、风险原因和可选命令分类结果。"""

    requires_approval: bool
    reason: str
    risk: str
    command_decision: CommandDecision | None = None


class ApprovalRequest(BaseModel):
    """Reviewer 决策所需的 Session、Turn、工具和风险快照。"""

    approval_id: str
    session_id: str
    turn_id: str
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    risk: str


class ApprovalDecision(BaseModel):
    """用户或自动 Reviewer 对一个 approval ID 的不可歧义决定。"""

    approval_id: str
    approved: bool
    reviewer: Literal["user", "auto"] = "auto"
    reason: str | None = None

    @classmethod
    def approve(cls, approval_id: str, reason: str | None = None) -> ApprovalDecision:
        """创建默认 reviewer 为 auto 的允许决定。

        Example:
            >>> ApprovalDecision.approve("appr_1").approved
            True
        """
        return cls(approval_id=approval_id, approved=True, reason=reason)

    @classmethod
    def deny(cls, approval_id: str, reason: str | None = None) -> ApprovalDecision:
        """创建默认 reviewer 为 auto 的拒绝决定。

        Example:
            >>> ApprovalDecision.deny("appr_1", "unsafe").reason
            'unsafe'
        """
        return cls(approval_id=approval_id, approved=False, reason=reason)


class ApprovalReviewer(ABC):
    """同步策略或交互式审批通道实现的两阶段审批接口。

    ``prepare`` 必须同步完成接收决定所需的登记；调用方只有在它返回后才能发布
    APPROVAL_REQUESTED。``review`` 随后等待或立即生成决定，``cancel`` 则覆盖
    事件发布失败、timeout 和 Turn 取消等所有提前退出路径。
    """

    @abstractmethod
    def prepare(self, request: ApprovalRequest) -> None:
        """在审批事件对外可见前同步建立接收决定所需的状态。"""
        ...

    @abstractmethod
    def cancel(self, request: ApprovalRequest) -> None:
        """幂等清理已准备但不再需要的审批状态。"""
        ...

    @abstractmethod
    async def review(self, request: ApprovalRequest) -> ApprovalDecision:
        """等待或立即返回与 ``request.approval_id`` 对应的决定。"""
        ...


class AutoApprovalReviewer(ApprovalReviewer):
    """按固定结果响应并记录请求的测试/显式自动 Reviewer。"""

    def __init__(self, *, approved: bool = True, reason: str | None = None) -> None:
        """配置所有后续请求共用的决定和原因。"""
        self.approved = approved
        self.reason = reason
        self.requests: list[ApprovalRequest] = []

    def prepare(self, request: ApprovalRequest) -> None:
        """自动 Reviewer 不接收外部决定，无需建立等待状态。"""

    def cancel(self, request: ApprovalRequest) -> None:
        """自动 Reviewer 没有待清理的交互状态。"""

    async def review(self, request: ApprovalRequest) -> ApprovalDecision:
        """记录请求并立即返回固定决定。"""
        self.requests.append(request)
        if self.approved:
            return ApprovalDecision.approve(request.approval_id, self.reason)
        return ApprovalDecision.deny(request.approval_id, self.reason)


class DenyApprovalReviewer(ApprovalReviewer):
    """未配置交互 Reviewer 时使用的 fail-closed 拒绝实现。"""

    def __init__(self) -> None:
        """创建空的请求审计列表。"""
        self.requests: list[ApprovalRequest] = []

    def prepare(self, request: ApprovalRequest) -> None:
        """固定拒绝 Reviewer 不接收外部决定，无需建立等待状态。"""

    def cancel(self, request: ApprovalRequest) -> None:
        """固定拒绝 Reviewer 没有待清理的交互状态。"""

    async def review(self, request: ApprovalRequest) -> ApprovalDecision:
        """记录请求并以稳定原因拒绝。"""
        self.requests.append(request)
        return ApprovalDecision.deny(
            request.approval_id,
            "no approval reviewer is configured",
        )


class ApprovalManager:
    """根据 approval policy 决定 tool call 是否需要用户确认。

    ``ApprovalManager`` 只负责风险评估、构造请求和调用 Reviewer，不执行工具，也
    不授予 Sandbox 能力。具体由谁批准取决于 Reviewer；CLI 场景通常使用
    ThreadApprovalReviewer，测试可以使用 AutoApprovalReviewer。未注入 Reviewer
    时默认拒绝，避免服务端或新入口无意中把交互审批降级成自动允许。
    """

    def __init__(
        self,
        *,
        reviewer: ApprovalReviewer | None = None,
        command_policy: CommandPolicy | None = None,
    ) -> None:
        """注入 Reviewer 和 shell 命令分类策略，默认无 Reviewer 时拒绝。"""
        self.reviewer = reviewer if reviewer is not None else DenyApprovalReviewer()
        self.command_policy = command_policy or CommandPolicy()

    async def evaluate(
        self,
        tool: BaseTool,
        call: ToolCall,
        args: BaseModel,
        context: TurnContext,
    ) -> ApprovalEvaluation:
        """结合审批策略、工具副作用和 shell 命令风险评估一次调用。

        Bash 始终先经过 ``CommandPolicy``；DENY 与 policy=NEVER 不创建审批，
        而把命令决定交给 BashTool 做硬拒绝、无交互拒绝或直接执行。这里的 NEVER
        表示“永不弹出审批”，不是“无条件允许”：PROMPT 命令在没有 approved 标记
        时仍被 BashTool 拒绝。其他工具在 UNTRUSTED 下按副作用审批，在 ON_REQUEST
        下按工具声明审批；不需要审批只代表治理链可以继续，不代表绕过 Sandbox。
        """
        if call.name == "bash":
            # shell command 的风险和 tool effect 不完全等价，需要交给 CommandPolicy。
            command = str(getattr(args, "command", ""))
            decision = self.command_policy.classify(
                command,
                network_access=context.network_access,
            )
            if (
                context.approval_policy == ApprovalPolicy.NEVER
                or decision.risk == CommandRisk.DENY
            ):
                return ApprovalEvaluation(
                    requires_approval=False,
                    reason=decision.reason,
                    risk=decision.risk,
                    command_decision=decision,
                )
            return ApprovalEvaluation(
                requires_approval=decision.requires_approval,
                reason=decision.reason,
                risk=decision.risk,
                command_decision=decision,
            )

        if context.approval_policy == ApprovalPolicy.NEVER:
            return ApprovalEvaluation(
                requires_approval=False,
                reason="approval disabled by policy",
                risk="safe",
            )

        if context.approval_policy == ApprovalPolicy.UNTRUSTED:
            write_effects = {
                ToolEffect.WORKSPACE_WRITE,
                ToolEffect.PROCESS_EXEC,
                ToolEffect.NETWORK,
                ToolEffect.EXTERNAL,
            }
            if tool.effects & write_effects:
                return ApprovalEvaluation(
                    requires_approval=True,
                    reason=f"{tool.name} has side effects",
                    risk="prompt",
                )

        if (
            context.approval_policy == ApprovalPolicy.ON_REQUEST
            and tool.requires_approval
        ):
            return ApprovalEvaluation(
                requires_approval=True,
                reason=f"{tool.name} requires approval",
                risk="prompt",
            )

        return ApprovalEvaluation(
            requires_approval=False,
            reason="tool is allowed",
            risk="safe",
        )

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """把请求交给 Reviewer，并用 approval ID 防止并发决定串单。

        Reviewer 返回错误 ID 属于审批通道失败，ToolRunner 会 fail-closed 为
        ``approval_error``，而不会把别的请求的允许决定应用到当前 call。
        """
        decision = await self.reviewer.review(request)
        if decision.approval_id != request.approval_id:
            raise RuntimeError("approval decision does not match its request")
        return decision

    def prepare(self, request: ApprovalRequest) -> None:
        """让 Reviewer 在 APPROVAL_REQUESTED 发布前同步进入可决定状态。"""
        self.reviewer.prepare(request)

    def cancel(self, request: ApprovalRequest) -> None:
        """幂等清理 Reviewer 为本次请求建立的等待状态。"""
        self.reviewer.cancel(request)

    @staticmethod
    def build_reviewer_failure_decision(
        request: ApprovalRequest,
        *,
        timed_out: bool,
    ) -> ApprovalDecision:
        """把审批超时或 Reviewer 异常 fail-closed 成自动拒绝决定。"""
        return ApprovalDecision.deny(
            request.approval_id,
            "approval timed out" if timed_out else "approval review failed",
        )

    @staticmethod
    def build_request(
        *,
        call: ToolCall,
        context: TurnContext,
        evaluation: ApprovalEvaluation,
    ) -> ApprovalRequest:
        """从调用、Turn 快照和风险评估创建唯一审批请求。

        Returns:
            带新 ``appr_`` ID 且复制调用参数和上下文标识的请求。
        """
        return ApprovalRequest(
            approval_id=new_id("appr_"),
            session_id=context.session_id,
            turn_id=context.turn_id,
            call_id=call.call_id,
            tool_name=call.name,
            arguments=call.arguments,
            reason=evaluation.reason,
            risk=evaluation.risk,
        )
