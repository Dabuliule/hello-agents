from __future__ import annotations

import json
from typing import Any

from codecraft.llm.base import LLMProtocolError


def get_field(value: Any, key: str, default: Any = None) -> Any:
    """从 SDK 对象或字典读取字段。

    Args:
        value: SDK 响应对象、字典或 ``None``。
        key: 要读取的字段名。
        default: 字段不存在时返回的值。

    Returns:
        字段值；字段不存在时返回 ``default``。

    Example:
        >>> get_field({"usage": {"input_tokens": 7}}, "usage")
        {'input_tokens': 7}
        >>> get_field(None, "usage", {})
        {}
    """
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def parse_arguments(value: Any) -> dict[str, Any]:
    """把供应商返回的工具参数解析成字典。

    Args:
        value: 已解析的字典，或者内容为 JSON 对象的字符串。

    Returns:
        可直接传给工具参数模型的字典。传入字典时返回原字典。

    Raises:
        LLMProtocolError: 输入为空、JSON 损坏，或 JSON 顶层不是对象。

    Example:
        >>> parse_arguments('{"path":"README.md"}')
        {'path': 'README.md'}
        >>> parse_arguments({"path": "README.md"})
        {'path': 'README.md'}
    """
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
    """把工具参数稳定序列化为供应商协议需要的紧凑 JSON。

    Args:
        arguments: 只包含 JSON 可序列化值的工具参数字典。

    Returns:
        保留 Unicode、移除多余空格并按键名排序的 JSON 字符串。

    Raises:
        TypeError: 参数中包含无法序列化为 JSON 的对象。

    Example:
        >>> serialize_arguments({"path": "你好.md", "line": 3})
        '{"line":3,"path":"你好.md"}'
    """
    return json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def required_string(value: Any, field: str) -> str:
    """校验工具调用身份字段为非空字符串。

    Args:
        value: 供应商返回的字段值。
        field: 用于错误消息的字段名，例如 ``call_id`` 或 ``name``。

    Returns:
        未经 ``strip`` 等改写的原字符串。

    Raises:
        LLMProtocolError: 值不是字符串，或字符串只包含空白。

    Example:
        >>> required_string("call_123", "call_id")
        'call_123'
    """
    if not isinstance(value, str) or not value.strip():
        raise LLMProtocolError(f"tool call requires a non-empty {field}")
    return value


def token_value(container: Any, field: str) -> int:
    """从 usage 对象或字典读取 token 计数。

    Args:
        container: SDK usage 对象或测试字典。
        field: token 字段名，例如 ``input_tokens``。

    Returns:
        非负整数；字段缺失或值为 ``None`` 时返回 ``0``。

    Raises:
        LLMProtocolError: 值是布尔值、负数或非整数。

    Example:
        >>> token_value({"input_tokens": 7}, "input_tokens")
        7
        >>> token_value({}, "cached_tokens")
        0
    """
    value = get_field(container, field, 0)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LLMProtocolError(f"usage.{field} must be a non-negative integer")
    return value


def error_message(value: Any, default: str) -> str:
    """从常见 SDK 错误结构中提取可读消息。

    Args:
        value: 原始 SDK 错误、响应对象或对应测试字典。
        default: 所有候选位置都没有消息时使用的文本。

    Returns:
        按 ``error.message``、``response.error.message``、根 ``message``
        顺序找到的第一个非空字符串，否则返回 ``default``。

    Example:
        >>> error_message({"error": {"message": "rate limited"}}, "failed")
        'rate limited'
        >>> error_message({}, "request failed")
        'request failed'
    """
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
