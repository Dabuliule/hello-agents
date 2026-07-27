from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelMessageType(StrEnum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ModelMessage(BaseModel):
    """Provider 无关的模型输入项。

    普通消息、工具调用和工具结果共用一个传输模型，但每种类型都有严格的
    字段组合。Provider 适配器可以据此直接转换协议，不需要猜测缺失字段。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ModelMessageType = ModelMessageType.MESSAGE
    role: ModelRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> ModelMessage:
        """保证每种输入项只有一种明确、完整的表示。"""
        if self.type == ModelMessageType.MESSAGE:
            self._validate_message()
        elif self.type == ModelMessageType.TOOL_CALL:
            self._validate_tool_call()
        else:
            self._validate_tool_result()
        return self

    def _validate_message(self) -> None:
        if self.role == ModelRole.TOOL:
            raise ValueError("ordinary messages cannot use the tool role")
        if not self.content:
            raise ValueError("ordinary messages require non-empty content")
        if any(
            value is not None
            for value in (self.name, self.tool_call_id, self.arguments)
        ):
            raise ValueError("ordinary messages cannot include tool fields")

    def _validate_tool_call(self) -> None:
        if self.role != ModelRole.ASSISTANT:
            raise ValueError("tool calls must use the assistant role")
        if self.content is not None:
            raise ValueError("tool calls cannot include text content")
        if not self.name or not self.name.strip():
            raise ValueError("tool calls require a name")
        if not self.tool_call_id or not self.tool_call_id.strip():
            raise ValueError("tool calls require a call id")
        if self.arguments is None:
            raise ValueError("tool calls require structured arguments")

    def _validate_tool_result(self) -> None:
        if self.role != ModelRole.TOOL:
            raise ValueError("tool results must use the tool role")
        if self.content is None:
            raise ValueError("tool results require content")
        if not self.tool_call_id or not self.tool_call_id.strip():
            raise ValueError("tool results require a call id")
        if self.name is not None or self.arguments is not None:
            raise ValueError("tool results cannot include call fields")
