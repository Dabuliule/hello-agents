from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import json
from typing import Any

from pydantic import BaseModel, Field

from codecraft.core.ids import new_id
from codecraft.llm.messages import ModelMessage, ModelMessageType, ModelRole
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
                messages.append(
                    ModelMessage(
                        type=ModelMessageType.TOOL_RESULT,
                        role=role,
                        content=item.content,
                        tool_call_id=item.tool_call_id,
                    )
                )
                continue

            if item.metadata.get("type") == ModelMessageType.TOOL_CALL.value:
                messages.append(
                    ModelMessage(
                        type=ModelMessageType.TOOL_CALL,
                        role=role,
                        name=item.name,
                        tool_call_id=item.tool_call_id,
                        arguments=item.arguments,
                    )
                )
                continue

            messages.append(ModelMessage(role=role, content=item.content))

        return messages

    def last_user_message(self) -> ConversationItem | None:
        for item in reversed(self.items):
            if item.role == ConversationRole.USER:
                return item
        return None

    def context_chars(self) -> int:
        """Return a provider-neutral size estimate for the model-visible history."""
        return len(
            json.dumps(
                [
                    message.model_dump(mode="json")
                    for message in self.build_model_messages()
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def compact(
        self,
        *,
        max_chars: int,
        keep_recent_items: int,
    ) -> dict[str, Any] | None:
        """Replace older complete turns with a deterministic summary.

        The latest user turn is always kept intact so function calls and their
        outputs cannot be separated. If that turn alone exceeds the budget, the
        caller must reject the request instead of producing an invalid history.
        """
        before_chars = self.context_chars()
        if before_chars <= max_chars or not self.items:
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

        overflow = compacted.context_chars() - max_chars
        if overflow > 0:
            shortened = summary_text[
                : max(0, len(summary_text) - overflow - 1)
            ].rstrip()
            compacted.items[0].content = shortened

        if not compacted.items[0].content or compacted.context_chars() > max_chars:
            return None

        self.items = compacted.items
        after_chars = self.context_chars()
        return {
            "summary": self.items[0].content,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "removed_items": len(removed),
            "retained_items": len(retained),
            "conversation": self.model_dump(mode="json"),
        }

    @staticmethod
    def _summarize(items: list[ConversationItem]) -> str:
        lines = ["Earlier conversation summary:"]
        for item in items:
            if item.metadata.get("type") == ModelMessageType.TOOL_CALL.value:
                detail = f"requested tool {item.name or 'unknown'}"
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
    def _to_model_role(role: ConversationRole) -> ModelRole | None:
        if role == ConversationRole.SUMMARY:
            return ModelRole.SYSTEM
        if role == ConversationRole.SYSTEM:
            return ModelRole.SYSTEM
        if role == ConversationRole.USER:
            return ModelRole.USER
        if role == ConversationRole.ASSISTANT:
            return ModelRole.ASSISTANT
        if role == ConversationRole.TOOL:
            return ModelRole.TOOL
        return None
