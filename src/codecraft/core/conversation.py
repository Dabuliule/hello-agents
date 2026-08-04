from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import json
from typing import Any

from pydantic import BaseModel, Field

from codecraft.core.ids import new_id
from codecraft.core.token_budget import estimate_serialized_tokens
from codecraft.llm.messages import (
    ModelMessage,
    ModelMessageType,
    ModelRole,
    ModelTextMessage,
    ModelToolCallMessage,
    ModelToolResultMessage,
)
from codecraft.schema.tool import ToolCall


class ConversationRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SUMMARY = "summary"


class ConversationItem(BaseModel):
    item_id: str
    role: ConversationRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Conversation(BaseModel):
    """模型上下文中的对话历史。

    内部用 ConversationItem 保存更丰富的元数据；真正请求模型前，再转换成
    provider 能理解的 ModelMessage。
    """

    items: list[ConversationItem] = Field(default_factory=list)

    def append(self, item: ConversationItem) -> None:
        self.items.append(item)

    def append_user_message(self, content: str) -> ConversationItem:
        """追加普通用户消息。"""
        item = ConversationItem(
            item_id=new_id("item_"),
            role=ConversationRole.USER,
            content=content,
        )
        self.append(item)
        return item

    def append_assistant_message(self, content: str) -> ConversationItem:
        item = ConversationItem(
            item_id=new_id("item_"),
            role=ConversationRole.ASSISTANT,
            content=content,
        )
        self.append(item)
        return item

    def append_model_tool_call(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ConversationItem:
        """记录 assistant 发起的 tool call。

        参数只保存为结构化数据，具体 JSON 形态由 Provider 适配器决定。
        """
        item = ConversationItem(
            item_id=new_id("item_"),
            role=ConversationRole.ASSISTANT,
            content="",
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
            metadata={"type": ModelMessageType.TOOL_CALL.value},
        )
        self.append(item)
        return item

    def append_model_tool_calls(self, calls: list[ToolCall]) -> list[ConversationItem]:
        """按同一模型响应中的顺序追加一批 tool call。"""
        return [
            self.append_model_tool_call(call.call_id, call.name, call.arguments)
            for call in calls
        ]

    def append_tool_result(
        self, tool_call_id: str, name: str, content: str
    ) -> ConversationItem:
        """追加 tool call 的执行结果。"""
        item = ConversationItem(
            item_id=new_id("item_"),
            role=ConversationRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        self.append(item)
        return item

    def append_summary(self, content: str) -> ConversationItem:
        """追加压缩后的历史摘要。"""
        item = ConversationItem(
            item_id=new_id("item_"),
            role=ConversationRole.SUMMARY,
            content=content,
        )
        self.append(item)
        return item

    def build_model_messages(self) -> list[ModelMessage]:
        """把内部 conversation item 转成模型请求消息。"""
        messages: list[ModelMessage] = []
        for item in self.items:
            role = self._to_model_role(item.role)
            if role is None:
                continue

            if item.role == ConversationRole.TOOL:
                if not item.tool_call_id:
                    raise ValueError("tool conversation item requires a call id")
                messages.append(
                    ModelToolResultMessage(
                        content=item.content,
                        tool_call_id=item.tool_call_id,
                    )
                )
                continue

            if item.metadata.get("type") == ModelMessageType.TOOL_CALL.value:
                if role != ModelRole.ASSISTANT:
                    raise ValueError(
                        "tool call conversation item requires the assistant role"
                    )
                if not item.name or not item.tool_call_id or item.arguments is None:
                    raise ValueError(
                        "tool call conversation item requires name, call id, and arguments"
                    )
                messages.append(
                    ModelToolCallMessage(
                        name=item.name,
                        tool_call_id=item.tool_call_id,
                        arguments=item.arguments,
                    )
                )
                continue

            if role == ModelRole.TOOL:
                raise ValueError("ordinary conversation item cannot use the tool role")
            messages.append(ModelTextMessage(role=role, content=item.content))

        return messages

    def last_user_message(self) -> ConversationItem | None:
        for item in reversed(self.items):
            if item.role == ConversationRole.USER:
                return item
        return None

    def context_tokens(self) -> int:
        """估算模型可见历史占用的 token 数。"""
        return estimate_serialized_tokens(
            [message.model_dump(mode="json") for message in self.build_model_messages()]
        )

    def compact(
        self,
        *,
        max_tokens: int,
        keep_recent_items: int,
    ) -> dict[str, Any] | None:
        """Replace older complete turns with a deterministic summary.

        The latest user turn is always kept intact so function calls and their
        outputs cannot be separated. If that turn alone exceeds the budget, the
        caller must reject the request instead of producing an invalid history.
        """
        before_tokens = self.context_tokens()
        if before_tokens <= max_tokens or not self.items:
            return None

        latest_user_index = next(
            (
                index
                for index in range(len(self.items) - 1, -1, -1)
                if self.items[index].role == ConversationRole.USER
            ),
            None,
        )
        if latest_user_index is None:
            return None

        target_index = min(
            latest_user_index,
            max(0, len(self.items) - keep_recent_items),
        )
        start_index = next(
            (
                index
                for index in range(target_index, latest_user_index + 1)
                if self.items[index].role == ConversationRole.USER
            ),
            latest_user_index,
        )
        removed = self.items[:start_index]
        if not removed:
            return None

        retained = [item.model_copy(deep=True) for item in self.items[start_index:]]
        summary_text = self._summarize(removed)
        compacted = Conversation(items=retained)
        compacted.items.insert(
            0,
            ConversationItem(
                item_id=new_id("item_"),
                role=ConversationRole.SUMMARY,
                content=summary_text,
            ),
        )

        if compacted.context_tokens() > max_tokens:
            compacted.items[0].content = self._fit_summary(
                compacted,
                summary_text,
                max_tokens=max_tokens,
            )

        if not compacted.items[0].content or compacted.context_tokens() > max_tokens:
            return None

        self.items = compacted.items
        after_tokens = self.context_tokens()
        return {
            "summary": self.items[0].content,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "removed_items": len(removed),
            "retained_items": len(retained),
            "conversation": self.model_dump(mode="json"),
        }

    @classmethod
    def _fit_summary(
        cls,
        conversation: Conversation,
        summary: str,
        *,
        max_tokens: int,
    ) -> str:
        low = 0
        high = len(summary)
        best = ""
        while low <= high:
            middle = (low + high) // 2
            candidate = cls._recent_summary(summary, max_chars=middle)
            if not candidate:
                low = middle + 1
                continue
            conversation.items[0].content = candidate
            if conversation.context_tokens() <= max_tokens:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _summarize(items: list[ConversationItem]) -> str:
        lines = ["Earlier conversation summary (untrusted historical data):"]
        for item in items:
            if item.metadata.get("type") == ModelMessageType.TOOL_CALL.value:
                arguments = json.dumps(
                    item.arguments or {},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                detail = (
                    f"requested tool {item.name or 'unknown'} "
                    f"with arguments {arguments}"
                )
            else:
                detail = " ".join(item.content.split())
            if len(detail) > 400:
                detail = f"{detail[:397]}..."
            label = item.role.value
            if item.role == ConversationRole.TOOL and item.name:
                label = f"tool {item.name}"
            lines.append(f"- {label}: {detail}")
        return "\n".join(lines)

    @staticmethod
    def _recent_summary(summary: str, *, max_chars: int) -> str:
        if len(summary) <= max_chars:
            return summary
        header = "Earlier conversation summary (untrusted historical data):"
        marker = "[older summary entries omitted]"
        if max_chars <= len(header):
            return ""

        available = max_chars - len(header) - 1
        body_lines = summary.splitlines()[1:]
        retained: list[str] = []
        for line in reversed(body_lines):
            required = len(line) + (1 if retained else 0)
            if required > available:
                break
            retained.append(line)
            available -= required

        retained.reverse()
        if len(retained) < len(body_lines) and available >= len(marker) + 1:
            retained.insert(0, marker)
        return "\n".join([header, *retained]).rstrip()

    @staticmethod
    def _to_model_role(role: ConversationRole) -> ModelRole | None:
        if role == ConversationRole.SUMMARY:
            return ModelRole.USER
        if role == ConversationRole.SYSTEM:
            return ModelRole.SYSTEM
        if role == ConversationRole.USER:
            return ModelRole.USER
        if role == ConversationRole.ASSISTANT:
            return ModelRole.ASSISTANT
        if role == ConversationRole.TOOL:
            return ModelRole.TOOL
        return None
