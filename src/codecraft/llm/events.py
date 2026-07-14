from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codecraft.schema.tool import ToolCall


class ModelEventType(StrEnum):
    MESSAGE_DELTA = "message_delta"
    MESSAGE_COMPLETED = "message_completed"
    TOOL_CALL = "tool_call"
    TOKEN_COUNT = "token_count"
    COMPLETED = "completed"


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


class ModelCompletedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


ModelEventPayload = (
    ModelTextPayload | ModelTokenCountPayload | ModelCompletedPayload | ToolCall
)


class ModelEvent(BaseModel):
    """Provider 向运行时暴露的统一成功事件。

    事件只表达模型输出数据和成功终止；调用失败通过 ``LLMProviderError``
    或 ``LLMProtocolError`` 表达。一次成功响应必须以 ``COMPLETED`` 结束。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ModelEventType
    payload: ModelEventPayload = Field(default_factory=ModelCompletedPayload)

    @model_validator(mode="before")
    @classmethod
    def validate_payload_for_type(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        event_type = ModelEventType(value.get("type"))
        payload_type: type[BaseModel]
        if event_type in {
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.MESSAGE_COMPLETED,
        }:
            payload_type = ModelTextPayload
        elif event_type == ModelEventType.TOOL_CALL:
            payload_type = ToolCall
        elif event_type == ModelEventType.TOKEN_COUNT:
            payload_type = ModelTokenCountPayload
        else:
            payload_type = ModelCompletedPayload

        normalized = dict(value)
        normalized["payload"] = payload_type.model_validate(value.get("payload", {}))
        return normalized
