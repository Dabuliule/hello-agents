from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """生成带领域前缀的 UUID4 hex 标识符。

    Example:
        ``new_id("turn_")`` 返回形如 ``turn_<32位hex>`` 的字符串。
    """
    return f"{prefix}{uuid4().hex}"
