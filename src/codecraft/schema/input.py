from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codecraft.schema.safety import sanitize_text


class SessionInputType(StrEnum):
    """进入 AgentThread 的数据输入与旁路控制输入类型。"""

    USER_MESSAGE = "user_message"
    INTERRUPT = "interrupt"
    APPROVAL_DECISION = "approval_decision"


class UserMessagePayload(BaseModel):
    """清洗后必须包含可见字符的用户消息。"""

    model_config = ConfigDict(extra="forbid")

    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """修复非法 Unicode，并拒绝清洗后只含空白的消息。"""
        sanitized = sanitize_text(value)
        if not sanitized.strip():
            raise ValueError("user message text must not be blank")
        return sanitized


class InterruptPayload(BaseModel):
    """请求取消当前 Turn 的清洗后原因。"""

    model_config = ConfigDict(extra="forbid")

    reason: str = "user_interrupt"

    @field_validator("reason")
    @classmethod
    def sanitize_reason(cls, value: str) -> str:
        """清洗中断原因；空结果回退为稳定的 ``user_interrupt``。"""
        return sanitize_text(value) or "user_interrupt"


class ApprovalDecisionPayload(BaseModel):
    """用户针对某个 approval ID 提交的允许或拒绝决定。"""

    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1)
    approved: bool
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def sanitize_reason(cls, value: str | None) -> str | None:
        """清洗可选审批说明，保留 ``None`` 语义。"""
        return sanitize_text(value) if value is not None else None


SessionInputPayload = UserMessagePayload | InterruptPayload | ApprovalDecisionPayload


class SessionInput(BaseModel):
    """带 ID、时间戳和按 type 严格分派 payload 的 Thread 输入。

    Example:
        >>> item = SessionInput.user_message("inp_1", "你好")
        >>> (item.type, item.payload.text)
        (<SessionInputType.USER_MESSAGE: 'user_message'>, '你好')
    """

    model_config = ConfigDict(extra="forbid")

    input_id: str = Field(min_length=1)
    type: SessionInputType
    payload: SessionInputPayload
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def validate_payload_for_type(cls, value: Any) -> Any:
        """根据输入 type 选择用户消息、中断或审批载荷模型。"""
        if not isinstance(value, dict):
            return value
        input_type = SessionInputType(cast(str, value.get("type")))
        payload_type: type[BaseModel]
        if input_type == SessionInputType.USER_MESSAGE:
            payload_type = UserMessagePayload
        elif input_type == SessionInputType.INTERRUPT:
            payload_type = InterruptPayload
        else:
            payload_type = ApprovalDecisionPayload
        normalized = dict(value)
        normalized["payload"] = payload_type.model_validate(value.get("payload", {}))
        return normalized

    @classmethod
    def user_message(cls, input_id: str, text: str) -> SessionInput:
        """创建一条经过 Unicode 清洗和空白校验的用户消息输入。"""
        return cls(
            input_id=input_id,
            type=SessionInputType.USER_MESSAGE,
            payload={"text": sanitize_text(text)},
        )

    @classmethod
    def approval_decision(
        cls,
        input_id: str,
        *,
        approval_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> SessionInput:
        """创建一条可旁路唤醒当前审批 Future 的用户决定输入。"""
        return cls(
            input_id=input_id,
            type=SessionInputType.APPROVAL_DECISION,
            payload={
                "approval_id": approval_id,
                "approved": approved,
                "reason": reason,
            },
        )
