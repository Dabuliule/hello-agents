from __future__ import annotations

import json
from math import ceil
from typing import Any


def estimate_text_tokens(text: str) -> int:
    """保守估算跨 Provider 文本 token 数。

    英文通常接近四字符一个 token，中文和代码符号的密度更高；同时使用字符
    数和 UTF-8 字节数，可以避免纯字符预算明显低估中文输入。
    """
    if not text:
        return 0
    return max(1, ceil(len(text) / 4), ceil(len(text.encode("utf-8")) / 3))


def estimate_serialized_tokens(value: Any) -> int:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return estimate_text_tokens(serialized)


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """返回不超过估算 token 上限的文本前缀。"""
    if max_tokens <= 0:
        return ""
    if estimate_text_tokens(text) <= max_tokens:
        return text

    low = 0
    high = len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]
