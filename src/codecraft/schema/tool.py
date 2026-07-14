from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codecraft.schema.event import RuntimeEventType


class ToolEffect(StrEnum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    PROCESS_EXEC = "process_exec"
    NETWORK = "network"
    EXTERNAL = "external"


class ToolSpec(BaseModel):
    """暴露给模型看的 tool 描述。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    effects: set[ToolEffect] = Field(default_factory=set)
    requires_approval: bool = False
    enabled: bool = True


class ToolCall(BaseModel):
    """模型请求执行某个 tool 时的结构化调用。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolRuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RuntimeEventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_tool_event_type(cls, value: RuntimeEventType) -> RuntimeEventType:
        if value != RuntimeEventType.PATCH_APPLIED:
            raise ValueError("tool results cannot emit runtime lifecycle events")
        return value


class ToolResult(BaseModel):
    """tool 执行完成后回传给模型和 UI 的结果。"""

    success: bool
    content: str
    data: dict[str, Any] | None = None
    error: str | None = None
    suggestion: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    runtime_events: list[ToolRuntimeEvent] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def validate_error_shape(self) -> ToolResult:
        """保持 success 和 error 字段语义一致。"""
        if self.success and self.error is not None:
            raise ValueError("successful tool results cannot include error")

        if not self.success and not self.error:
            raise ValueError("failed tool results must include error")

        return self

    def model_content(self) -> str:
        """返回包含失败和截断语义的模型可见文本。"""
        details = [self.content]
        if self.error is not None:
            details.append(f"[tool_error: {self.error}]")
        if self.suggestion:
            details.append(f"[suggestion: {self.suggestion}]")
        if self.metadata.get("content_truncated") is True:
            original_chars = self.metadata.get("original_content_chars", "unknown")
            details.append(
                f"[output truncated from {original_chars} characters; request a narrower result]"
            )
        if self.data is not None and self.data.get("data_truncated") is True:
            original_chars = self.data.get("original_data_chars", "unknown")
            details.append(
                f"[structured data truncated from {original_chars} characters]"
            )
        if self.metadata.get("metadata_truncated") is True:
            details.append("[tool metadata truncated]")
        return "\n".join(part for part in details if part)
