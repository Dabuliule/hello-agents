from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|cookie|credential|password|secret|token)([_-]|$)",
    re.IGNORECASE,
)
REDACTED = "[REDACTED]"


def sanitize_text(value: str) -> str:
    """把无法编码的 Unicode surrogate 替换为安全字符。

    Example:
        >>> sanitize_text("ok" + chr(0xD800))
        'ok?'
    """
    return value.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_json_value(value: Any) -> Any:
    """递归清洗 JSON 风格容器的键和值，并把 tuple 规范为 list。

    Example:
        >>> sanitize_json_value({"items": ("a", "b")})
        {'items': ['a', 'b']}
    """
    if isinstance(value, str):
        return sanitize_text(value)

    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, Mapping):
        return {
            sanitize_text(str(key)): sanitize_json_value(item)
            for key, item in value.items()
        }

    return value


def redact_sensitive_json_value(value: Any) -> Any:
    """按字段名递归脱敏持久化事件中的常见凭据。"""
    if isinstance(value, list):
        return [redact_sensitive_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [redact_sensitive_json_value(item) for item in value]

    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            key_text = sanitize_text(str(key))
            redacted[key_text] = (
                REDACTED
                if _is_sensitive_key(key_text)
                else redact_sensitive_json_value(item)
            )
        return redacted

    return value


def _is_sensitive_key(key: str) -> bool:
    """判断字段名是否表示凭据，同时保留环境变量名等描述性字段。"""
    normalized = key.lower()
    if normalized.endswith(("_env", "_field", "_name")):
        return False
    return _SENSITIVE_KEY.search(normalized) is not None
