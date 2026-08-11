from __future__ import annotations

from typing import Any, Protocol

from codecraft.core.turn_context import TurnContext
from codecraft.schema.tool import ToolCall, ToolResult


class ToolResultObserver(Protocol):
    """成功工具结果后的非关键异步副作用扩展点。"""

    name: str

    async def after_result(
        self,
        call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> dict[str, Any] | None:
        """观察调用和结果；返回可附入 post_actions 的诊断详情。"""
        ...
