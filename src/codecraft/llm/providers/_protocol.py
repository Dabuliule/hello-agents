from __future__ import annotations

import json
from typing import Any

from codecraft.llm.base import LLMProtocolError


def get_field(value: Any, key: str, default: Any = None) -> Any:
    """读取 SDK object 或测试字典中的同名字段。"""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_arguments(value: Any) -> dict[str, Any]:
    """严格解析模型生成的工具参数，禁止用空对象掩盖损坏的 JSON。"""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        raise LLMProtocolError("tool call arguments must be a JSON object")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError("tool call arguments contain invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMProtocolError("tool call arguments must decode to an object")
    return parsed


def serialize_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LLMProtocolError(f"tool call requires a non-empty {field}")
    return value


def token_value(container: Any, field: str) -> int:
    value = get_field(container, field, 0)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LLMProtocolError(f"usage.{field} must be a non-negative integer")
    return value


def error_message(value: Any, default: str) -> str:
    candidates = [
        get_field(value, "error"),
        get_field(get_field(value, "response"), "error"),
        value,
    ]
    for candidate in candidates:
        message = get_field(candidate, "message")
        if isinstance(message, str) and message:
            return message
    return default
