from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from codecraft.schema.safety import redact_sensitive_json_value, sanitize_json_value

RUNTIME_EVENT_SCHEMA_VERSION: Literal[1] = 1


class RuntimeEventType(StrEnum):
    """Runtime 持久化事件的稳定类型集合。"""

    SESSION_STARTED = "session_started"
    SESSION_RESTORED = "session_restored"

    TURN_STARTED = "turn_started"
    USER_MESSAGE = "user_message"

    ASSISTANT_MESSAGE_DELTA = "assistant_message_delta"
    ASSISTANT_MESSAGE = "assistant_message"

    MODEL_TOOL_CALL = "model_tool_call"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"

    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"

    PATCH_APPLIED = "patch_applied"
    TOKEN_COUNT = "token_count"
    CONTEXT_COMPACTED = "context_compacted"

    ERROR = "error"
    TURN_FINISHED = "turn_finished"
    TURN_ABORTED = "turn_aborted"
    SESSION_CLOSED = "session_closed"


class EventPayload(BaseModel):
    """持久化事件共用的不可变、拒绝额外字段且兼容映射访问的载荷。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __getitem__(self, key: str) -> Any:
        """按字段名读取载荷，未知字段抛 ``KeyError``。"""
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """按映射语义读取字段，未知字段返回 ``default``。"""
        try:
            return self[key]
        except KeyError:
            return default

    def __str__(self) -> str:
        """返回仅含显式字段的 Python 字典文本，便于日志展示。"""
        return str(self.model_dump(mode="python", exclude_unset=True))


class EmptyEventPayload(EventPayload):
    """不携带业务数据的 session closed 等事件载荷。"""


class SessionStartedEventPayload(EventPayload):
    """Session 首事件中的完整配置快照和可选 Skill 快照。"""

    config: dict[str, Any]
    skills: dict[str, Any] | None = None

    @field_validator("config")
    @classmethod
    def validate_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        """通过 ``SessionConfig`` 校验并规范化可恢复配置快照。"""
        from codecraft.schema.session import SessionConfig

        return SessionConfig.model_validate(value).model_dump(mode="json")

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """校验可用 Skill 与诊断信息的启动快照。"""
        return _validate_skill_snapshot(value)


class SessionRestoredEventPayload(EventPayload):
    """Session 恢复时重新发现的可选 Skill 快照。"""

    skills: dict[str, Any] | None = None

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """校验恢复事件中的 Skill 快照。"""
        return _validate_skill_snapshot(value)


class TurnStartedEventPayload(EventPayload):
    """记录新 Turn 所消费的输入 ID。"""

    input_id: str = Field(min_length=1)


class UserMessageEventPayload(EventPayload):
    """持久化用户输入 ID 与非空文本。"""

    input_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class TextEventPayload(EventPayload):
    """assistant 增量或完整消息的非空文本载荷。"""

    text: str = Field(min_length=1)


class ToolCallEventPayload(EventPayload):
    """模型意图或工具开始事件共用的结构化调用载荷。"""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ToolTimingsEventPayload(BaseModel):
    """工具治理、审批等待、执行、观察器和总耗时的毫秒拆分。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    governance: int = Field(ge=0)
    approval_wait: int = Field(ge=0)
    execution: int = Field(ge=0)
    observers: int = Field(ge=0)
    total: int = Field(ge=0)


class ToolCallFinishedEventPayload(EventPayload):
    """工具结果、总耗时和可选阶段耗时。"""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    result: dict[str, Any]
    duration_ms: int = Field(ge=0)
    timings_ms: dict[str, Any] | None = None

    @field_validator("result")
    @classmethod
    def validate_result(cls, value: dict[str, Any]) -> dict[str, Any]:
        """通过 ``ToolResult`` 校验并规范化持久化结果。"""
        from codecraft.schema.tool import ToolResult

        return ToolResult.model_validate(value).model_dump(mode="json")

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """校验可选的工具阶段耗时字典。"""
        if value is None:
            return None
        return ToolTimingsEventPayload.model_validate(value).model_dump(mode="json")


class ApprovalRequestedEventPayload(EventPayload):
    """向用户展示并持久化的一次工具审批请求。"""

    approval_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    reason: str = Field(min_length=1)
    risk: str = Field(min_length=1)


class ApprovalDecidedEventPayload(EventPayload):
    """用户或自动 Reviewer 对某个审批请求的决定。"""

    approval_id: str = Field(min_length=1)
    approved: bool
    reviewer: Literal["user", "auto"] = "auto"
    reason: str | None = None


class PatchAppliedEventPayload(EventPayload):
    """完整补丁统计或因过大而截断的互斥载荷。"""

    call_id: str = Field(min_length=1)
    changed_files: list[str] | None = None
    modified: int | None = Field(default=None, ge=0)
    added: int | None = Field(default=None, ge=0)
    deleted: int | None = Field(default=None, ge=0)
    payload_truncated: Literal[True] | None = None
    original_payload_chars: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> PatchAppliedEventPayload:
        """确保完整统计形态与截断占位形态互斥且字段完备。"""
        if self.payload_truncated is True:
            if self.original_payload_chars is None:
                raise ValueError(
                    "truncated patch payload requires original_payload_chars"
                )
            if any(
                value is not None
                for value in (
                    self.changed_files,
                    self.modified,
                    self.added,
                    self.deleted,
                )
            ):
                raise ValueError("truncated patch payload cannot include patch counts")
            return self

        if self.original_payload_chars is not None:
            raise ValueError("original_payload_chars requires payload_truncated=true")
        if any(
            value is None
            for value in (
                self.changed_files,
                self.modified,
                self.added,
                self.deleted,
            )
        ):
            raise ValueError("patch payload requires changed files and patch counts")
        return self


class TokenCountEventPayload(EventPayload):
    """Runtime 持久化的统一 Token 用量，reasoning/cached 均为子集。"""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def fill_total_tokens(cls, value: Any) -> Any:
        """显式 total 缺失时，用 input 与 output 之和补齐浅拷贝。"""
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
    def validate_total_tokens(self) -> TokenCountEventPayload:
        """确认 total 没有重复加上 reasoning 或 cached 子集。"""
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class ContextCompactedEventPayload(EventPayload):
    """上下文压缩前后指标、摘要和可恢复 Conversation 快照。"""

    summary: str = Field(min_length=1)
    before_tokens: int = Field(ge=0)
    after_tokens: int = Field(ge=0)
    removed_items: int = Field(ge=1)
    retained_items: int = Field(ge=0)
    conversation: dict[str, Any]

    @field_validator("conversation")
    @classmethod
    def validate_conversation(cls, value: dict[str, Any]) -> dict[str, Any]:
        """通过 ``Conversation`` 校验压缩后的可恢复快照。"""
        from codecraft.core.conversation import Conversation

        return Conversation.model_validate(value).model_dump(mode="json")


class ErrorEventPayload(EventPayload):
    """稳定错误码、可读消息、元数据和可选修复建议。"""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    metadata: dict[str, Any]
    suggestion: str | None = None


class TurnFinishedEventPayload(EventPayload):
    """成功 Turn 的最终答案、工具次数和耗时。"""

    answer: str = Field(min_length=1)
    tool_calls: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class TurnAbortedEventPayload(EventPayload):
    """中止 Turn 的结构化原因、指标和诊断元数据。"""

    reason: str = Field(min_length=1)
    message: str = Field(min_length=1)
    tool_calls: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    metadata: dict[str, Any]


RuntimeEventPayload = (
    EmptyEventPayload
    | SessionStartedEventPayload
    | SessionRestoredEventPayload
    | TurnStartedEventPayload
    | UserMessageEventPayload
    | TextEventPayload
    | ToolCallEventPayload
    | ToolCallFinishedEventPayload
    | ApprovalRequestedEventPayload
    | ApprovalDecidedEventPayload
    | PatchAppliedEventPayload
    | TokenCountEventPayload
    | ContextCompactedEventPayload
    | ErrorEventPayload
    | TurnFinishedEventPayload
    | TurnAbortedEventPayload
)

_PAYLOAD_MODELS: dict[RuntimeEventType, type[EventPayload]] = {
    RuntimeEventType.SESSION_STARTED: SessionStartedEventPayload,
    RuntimeEventType.SESSION_RESTORED: SessionRestoredEventPayload,
    RuntimeEventType.TURN_STARTED: TurnStartedEventPayload,
    RuntimeEventType.USER_MESSAGE: UserMessageEventPayload,
    RuntimeEventType.ASSISTANT_MESSAGE_DELTA: TextEventPayload,
    RuntimeEventType.ASSISTANT_MESSAGE: TextEventPayload,
    RuntimeEventType.MODEL_TOOL_CALL: ToolCallEventPayload,
    RuntimeEventType.TOOL_CALL_STARTED: ToolCallEventPayload,
    RuntimeEventType.TOOL_CALL_FINISHED: ToolCallFinishedEventPayload,
    RuntimeEventType.APPROVAL_REQUESTED: ApprovalRequestedEventPayload,
    RuntimeEventType.APPROVAL_DECIDED: ApprovalDecidedEventPayload,
    RuntimeEventType.PATCH_APPLIED: PatchAppliedEventPayload,
    RuntimeEventType.TOKEN_COUNT: TokenCountEventPayload,
    RuntimeEventType.CONTEXT_COMPACTED: ContextCompactedEventPayload,
    RuntimeEventType.ERROR: ErrorEventPayload,
    RuntimeEventType.TURN_FINISHED: TurnFinishedEventPayload,
    RuntimeEventType.TURN_ABORTED: TurnAbortedEventPayload,
    RuntimeEventType.SESSION_CLOSED: EmptyEventPayload,
}


class RuntimeEvent(BaseModel):
    """可持久化、可恢复且按 seq 排序的不可变 Runtime 事实。

    Example:
        >>> event = RuntimeEvent(
        ...     event_id="evt_1",
        ...     session_id="ses_1",
        ...     seq=1,
        ...     type="assistant_message",
        ...     payload={"text": "done"},
        ... )
        >>> (event.type, event.payload["text"])
        (<RuntimeEventType.ASSISTANT_MESSAGE: 'assistant_message'>, 'done')
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = RUNTIME_EVENT_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    turn_id: str | None = Field(default=None, min_length=1)
    seq: int = Field(ge=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: RuntimeEventType
    payload: RuntimeEventPayload = Field(default_factory=EmptyEventPayload)

    @model_validator(mode="before")
    @classmethod
    def validate_payload_for_type(cls, value: Any) -> Any:
        """按 event type 选择唯一载荷模型，并在持久化前清洗和脱敏。

        Args:
            value: Pydantic 校验前收到的事件对象或原始字典。

        Returns:
            字典输入会复制并把 payload 替换成对应的严格模型；其他输入保持原样。

        Raises:
            ValueError: payload 不是 JSON object，或具体载荷校验失败。
        """
        if not isinstance(value, dict):
            return value
        event_type_value = value.get("type")
        if not isinstance(event_type_value, (str, RuntimeEventType)):
            return value
        try:
            event_type = RuntimeEventType(event_type_value)
        except (TypeError, ValueError):
            return value
        raw_payload = value.get("payload", {})
        if not isinstance(raw_payload, (dict, EventPayload)):
            raise ValueError("event payload must be a JSON object")
        payload = _sanitize_payload(
            raw_payload,
            redact=event_type != RuntimeEventType.SESSION_STARTED,
        )
        normalized = dict(value)
        normalized["payload"] = _PAYLOAD_MODELS[event_type].model_validate(payload)
        return normalized

    @field_serializer("payload")
    def serialize_payload(
        self,
        value: RuntimeEventPayload,
        info: SerializationInfo,
    ) -> dict[str, Any]:
        """让联合载荷按具体模型序列化，避免 Pydantic 丢失分支字段。"""
        return value.model_dump(
            mode=info.mode,
            include=cast(Any, info.include),
            exclude=cast(Any, info.exclude),
            context=info.context,
            by_alias=info.by_alias,
            exclude_unset=True,
            exclude_defaults=info.exclude_defaults,
            exclude_none=info.exclude_none,
            exclude_computed_fields=info.exclude_computed_fields,
            round_trip=info.round_trip,
            serialize_as_any=info.serialize_as_any,
        )


def _sanitize_payload(value: Any, *, redact: bool = True) -> dict[str, Any]:
    """把任意载荷规范为 JSON 兼容字典并按需递归脱敏。

    Args:
        value: ``EventPayload`` 或待清洗的原始对象。
        redact: 是否按敏感字段名替换凭据；SessionStarted 配置快照会关闭此步。

    Returns:
        清洗后的字典；顶层无法表示为字典时返回空字典。

    Example:
        >>> _sanitize_payload({"api_key": "secret", "count": 1})
        {'api_key': '[REDACTED]', 'count': 1}
    """
    if isinstance(value, EventPayload):
        value = value.model_dump(mode="python", exclude_unset=True)
    sanitized = sanitize_json_value(value)
    if isinstance(sanitized, dict):
        if not redact:
            return {str(key): item for key, item in sanitized.items()}
        redacted = redact_sensitive_json_value(sanitized)
        if isinstance(redacted, dict):
            return {str(key): item for key, item in redacted.items()}
    return {}


def _validate_skill_snapshot(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """严格校验事件中的 Skill metadata/diagnostic 快照。

    Args:
        value: ``None`` 或只含 available、diagnostics 两个列表的字典。

    Returns:
        ``None`` 或经具体 Skill 模型规范化的 JSON 字典。

    Raises:
        ValueError: 字段集合或列表形态不合法。
    """
    if value is None:
        return None
    from codecraft.skill.models import SkillDiagnostic, SkillMetadata

    available = value.get("available")
    diagnostics = value.get("diagnostics")
    if not isinstance(available, list) or not isinstance(diagnostics, list):
        raise ValueError("skill snapshot requires available and diagnostics lists")
    if set(value) != {"available", "diagnostics"}:
        raise ValueError("skill snapshot contains unsupported fields")
    return {
        "available": [
            SkillMetadata.model_validate(item).model_dump(mode="json")
            for item in available
        ],
        "diagnostics": [
            SkillDiagnostic.model_validate(item).model_dump(mode="json")
            for item in diagnostics
        ],
    }
