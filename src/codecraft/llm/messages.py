from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelRole(StrEnum):
    """文本消息角色；工具调用和结果的角色由具体消息类固定。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class _ModelMessageBase(BaseModel):
    """供应商无关、拒绝额外字段且不可变的模型输入项。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelTextMessage(_ModelMessageBase):
    """普通 system、user 或 assistant 非空文本消息。"""

    type: Literal["message"] = "message"
    role: Literal[ModelRole.SYSTEM, ModelRole.USER, ModelRole.ASSISTANT]
    content: str = Field(min_length=1)


class ModelToolCallMessage(_ModelMessageBase):
    """assistant 产生的结构化工具调用。"""

    type: Literal["tool_call"] = "tool_call"
    role: Literal[ModelRole.ASSISTANT] = ModelRole.ASSISTANT
    name: str
    tool_call_id: str
    arguments: dict[str, Any]

    @field_validator("name", "tool_call_id")
    @classmethod
    def validate_required_string(cls, value: str) -> str:
        """校验工具名或 call ID 至少包含一个非空白字符。

        Args:
            value: Pydantic 正在校验的名称或 call ID。

        Returns:
            保留原始空白的输入字符串。

        Raises:
            ValueError: 字符串为空或只含空白。

        Example:
            >>> ModelToolCallMessage.validate_required_string("call_read")
            'call_read'
        """
        if not value.strip():
            raise ValueError("tool calls require non-empty names and call ids")
        return value


class ModelToolResultMessage(_ModelMessageBase):
    """通过 call ID 关联某个先前工具调用的执行结果。"""

    type: Literal["tool_result"] = "tool_result"
    role: Literal[ModelRole.TOOL] = ModelRole.TOOL
    content: str
    tool_call_id: str

    @field_validator("tool_call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        """校验工具结果必须关联一个非空 call ID。

        Args:
            value: 待校验的 call ID。

        Returns:
            未改写的原字符串。

        Raises:
            ValueError: call ID 为空或只含空白。

        Example:
            >>> ModelToolResultMessage.validate_call_id("call_read")
            'call_read'
        """
        if not value.strip():
            raise ValueError("tool results require a non-empty call id")
        return value


ModelMessage = Annotated[
    ModelTextMessage | ModelToolCallMessage | ModelToolResultMessage,
    Field(discriminator="type"),
]
