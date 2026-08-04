from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _ModelMessageBase(BaseModel):
    """Provider-neutral, immutable model input item."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelTextMessage(_ModelMessageBase):
    """A normal system, user, or assistant text message."""

    type: Literal["message"] = "message"
    role: Literal[ModelRole.SYSTEM, ModelRole.USER, ModelRole.ASSISTANT]
    content: str = Field(min_length=1)


class ModelToolCallMessage(_ModelMessageBase):
    """A structured tool call emitted by the assistant."""

    type: Literal["tool_call"] = "tool_call"
    role: Literal[ModelRole.ASSISTANT] = ModelRole.ASSISTANT
    name: str
    tool_call_id: str
    arguments: dict[str, Any]

    @field_validator("name", "tool_call_id")
    @classmethod
    def validate_required_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool calls require non-empty names and call ids")
        return value


class ModelToolResultMessage(_ModelMessageBase):
    """The result associated with one earlier tool call."""

    type: Literal["tool_result"] = "tool_result"
    role: Literal[ModelRole.TOOL] = ModelRole.TOOL
    content: str
    tool_call_id: str

    @field_validator("tool_call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool results require a non-empty call id")
        return value


ModelMessage = Annotated[
    ModelTextMessage | ModelToolCallMessage | ModelToolResultMessage,
    Field(discriminator="type"),
]
