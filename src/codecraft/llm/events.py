from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codecraft.schema.tool import ToolCall


class ModelTextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)


class ModelTokenCountPayload(BaseModel):
    """一次模型调用的 Token 用量，reasoning 是 output 的子集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def fill_total_tokens(cls, value: Any) -> Any:
        if isinstance(value, dict) and "total_tokens" not in value:
            normalized = dict(value)
            input_tokens = normalized.get("input_tokens", 0)
            output_tokens = normalized.get("output_tokens", 0)
            if (
                isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
            ):
                normalized["total_tokens"] = input_tokens + output_tokens
            return normalized
        return value

    @model_validator(mode="after")
    def validate_total_tokens(self) -> ModelTokenCountPayload:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class _ModelEventBase(BaseModel):
    """Provider-to-runtime success event; failures are raised as exceptions."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelMessageDeltaEvent(_ModelEventBase):
    type: Literal["message_delta"] = "message_delta"
    payload: ModelTextPayload


class ModelMessageCompletedEvent(_ModelEventBase):
    type: Literal["message_completed"] = "message_completed"
    payload: ModelTextPayload


class ModelToolCallEvent(_ModelEventBase):
    type: Literal["tool_call"] = "tool_call"
    payload: ToolCall


class ModelTokenCountEvent(_ModelEventBase):
    type: Literal["token_count"] = "token_count"
    payload: ModelTokenCountPayload


class ModelCompletedEvent(_ModelEventBase):
    """Successful terminal marker for one provider response stream."""

    type: Literal["completed"] = "completed"


ModelEvent = Annotated[
    ModelMessageDeltaEvent
    | ModelMessageCompletedEvent
    | ModelToolCallEvent
    | ModelTokenCountEvent
    | ModelCompletedEvent,
    Field(discriminator="type"),
]
