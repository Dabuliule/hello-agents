from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codecraft.schema.safety import sanitize_text


class SessionInputType(StrEnum):
    USER_MESSAGE = "user_message"
    INTERRUPT = "interrupt"
    APPROVAL_DECISION = "approval_decision"


class UserMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        sanitized = sanitize_text(value)
        if not sanitized.strip():
            raise ValueError("user message text must not be blank")
        return sanitized


class InterruptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = "user_interrupt"

    @field_validator("reason")
    @classmethod
    def sanitize_reason(cls, value: str) -> str:
        return sanitize_text(value) or "user_interrupt"


class ApprovalDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(min_length=1)
    approved: bool
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def sanitize_reason(cls, value: str | None) -> str | None:
        return sanitize_text(value) if value is not None else None


SessionInputPayload = UserMessagePayload | InterruptPayload | ApprovalDecisionPayload


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str = Field(min_length=1)
    type: SessionInputType
    payload: SessionInputPayload
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def validate_payload_for_type(cls, value: Any) -> Any:
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
        return cls(
            input_id=input_id,
            type=SessionInputType.APPROVAL_DECISION,
            payload={
                "approval_id": approval_id,
                "approved": approved,
                "reason": reason,
            },
        )
