from __future__ import annotations

import json
from typing import Any

from codecraft.llm.base import LLMProtocolError


def get_field(value: Any, key: str, default: Any = None) -> Any:
    """从 SDK 对象或字典读取字段；字段不存在时返回 ``default``。"""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_arguments(value: Any) -> dict[str, Any]:
    """把字典或 JSON 字符串解析成工具参数对象，非法输入抛协议错误。"""
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
    """把工具参数稳定序列化为紧凑 JSON，供模型供应商协议发送。"""
    return json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def required_string(value: Any, field: str) -> str:
    """校验工具调用身份字段为非空字符串，并返回未经改写的原值。"""
    if not isinstance(value, str) or not value.strip():
        raise LLMProtocolError(f"tool call requires a non-empty {field}")
    return value


def token_value(container: Any, field: str) -> int:
    """读取可选的 token 计数字段；缺失值按零处理，非法数值抛协议错误。"""
    value = get_field(container, field, 0)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LLMProtocolError(f"usage.{field} must be a non-negative integer")
    return value


def error_message(value: Any, default: str) -> str:
    """从常见 SDK 错误嵌套结构提取消息，未找到时返回默认文本。"""
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
