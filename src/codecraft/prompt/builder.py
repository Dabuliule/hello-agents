from __future__ import annotations

from codecraft.core.conversation import Conversation
from codecraft.core.turn_context import TurnContext
from codecraft.llm.messages import ModelMessage, ModelRole
from codecraft.prompt.base_instructions import BASE_INSTRUCTIONS
from codecraft.schema.session import SessionConfig


class PromptBuilder:
    """组装发送给模型的 system prompt 和历史消息。"""

    def build(
        self,
        *,
        config: SessionConfig,
        conversation: Conversation,
        context: TurnContext,
        project_instructions: str | None = None,
        available_skills: str | None = None,
        active_skills: str | None = None,
    ) -> list[ModelMessage]:
        """按固定 section 顺序构造完整模型输入。"""
        sections = [
            ("base_instructions", config.base_instructions or BASE_INSTRUCTIONS),
            ("project_instructions", project_instructions),
            ("user_instructions", config.user_instructions),
            ("available_skills", available_skills),
            ("active_skills", active_skills),
            ("turn_context", self._turn_context(context)),
        ]
        content = "\n\n".join(
            f"<{name}>\n{body.strip()}\n</{name}>"
            for name, body in sections
            if body and body.strip()
        )
        return [
            ModelMessage(role=ModelRole.SYSTEM, content=content),
            *conversation.build_model_messages(),
        ]

    @staticmethod
    def _turn_context(context: TurnContext) -> str:
        return "\n".join(
            [
                f"cwd: {context.cwd}",
                f"workspace_roots: {', '.join(str(root) for root in context.workspace_roots)}",
                f"approval_policy: {context.approval_policy}",
                f"sandbox_mode: {context.sandbox_mode}",
                f"network_access: {str(context.network_access).lower()}",
            ]
        )
