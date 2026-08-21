from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from codecraft.schema.tool import ToolCall


class ModelTextPayload(BaseModel):
    """不可变的非空模型文本载荷。"""

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
        """在调用方省略 total 时用 input 与 output 之和补齐。

        Args:
            value: Pydantic 校验前收到的原始载荷。

        Returns:
            对字典输入返回补齐后的浅拷贝；其他输入保持不变。

        Example:
            >>> ModelTokenCountPayload(
            ...     input_tokens=7, output_tokens=3
            ... ).total_tokens
            10
        """
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
        """确认 total 等于 input 与 output 之和。

        Returns:
            校验成功的当前不可变载荷。

        Raises:
            ValueError: total 使用了与内部口径不一致的值。
        """
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class _ModelEventBase(BaseModel):
    """Provider 到 Runtime 的不可变成功事件；失败统一使用异常。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelMessageDeltaEvent(_ModelEventBase):
    """模型产生的一个有序非空文本 part。

    流式上游可以产生多个小 part；非流式完整响应由 Provider adapter 转换为一个
    大 part。消费者必须等 ``ModelCompletedEvent`` 后才能把累积文本视为成功结果。
    """

    type: Literal["message_delta"] = "message_delta"
    payload: ModelTextPayload


class ModelToolCallEvent(_ModelEventBase):
    """模型请求 Runtime 执行的一个结构化工具调用。"""

    type: Literal["tool_call"] = "tool_call"
    payload: ToolCall


class ModelTokenCountEvent(_ModelEventBase):
    """一次供应商响应确认后的统一 Token 用量。"""

    type: Literal["token_count"] = "token_count"
    payload: ModelTokenCountPayload


class ModelCompletedEvent(_ModelEventBase):
    """一次 Provider 响应流成功闭合的无载荷终止标志。"""

    type: Literal["completed"] = "completed"


ModelEvent = Annotated[
    ModelMessageDeltaEvent
    | ModelToolCallEvent
    | ModelTokenCountEvent
    | ModelCompletedEvent,
    Field(discriminator="type"),
]
