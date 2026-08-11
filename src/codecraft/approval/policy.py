from __future__ import annotations

from enum import StrEnum


class ApprovalPolicy(StrEnum):
    """禁用审批、按工具声明审批或对所有有副作用工具审批的策略。"""

    NEVER = "never"
    ON_REQUEST = "on_request"
    UNTRUSTED = "untrusted"
