from __future__ import annotations

from codecraft.core.conversation import Conversation
from codecraft.core.token_budget import estimate_text_tokens
from codecraft.core.turn_context import TurnContext
from codecraft.llm.messages import ModelMessage, ModelRole, ModelTextMessage
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
        """按固定 section 顺序构造 system 消息并追加 Conversation 历史。

        Args:
            config: Session 的基础/用户指令设置。
            conversation: 已持久化的文本、工具调用、结果和摘要历史。
            context: 当前 Turn 的 cwd、审批、沙箱和网络快照。
            project_instructions: 当前访问路径作用域内的项目规则。
            available_skills: 轻量 Skill 目录。
            active_skills: 本 Turn 已激活 Skill 的完整正文。

        Returns:
            第一项固定为 system prompt，后续保持 Conversation 顺序的模型消息。

        Example:
            ``builder.build(config=..., conversation=..., context=...)`` 返回
            ``[ModelTextMessage(role="system", ...), *history]``。
        """
        sections = self._sections(
            config=config,
            context=context,
            project_instructions=project_instructions,
            available_skills=available_skills,
            active_skills=active_skills,
        )
        content = "\n\n".join(
            self._render_section(name, body)
            for name, body in sections
            if body and body.strip()
        )
        return [
            ModelTextMessage(role=ModelRole.SYSTEM, content=content),
            *conversation.build_model_messages(),
        ]

    def fixed_section_tokens(
        self,
        *,
        config: SessionConfig,
        context: TurnContext,
        project_instructions: str | None = None,
        available_skills: str | None = None,
        active_skills: str | None = None,
    ) -> dict[str, int]:
        """估算各非空固定 section 连同标签的 Token，供超限诊断。

        Returns:
            section 名到估算 Token 数的映射；Conversation 历史不包含在内。
        """
        return {
            name: estimate_text_tokens(self._render_section(name, body))
            for name, body in self._sections(
                config=config,
                context=context,
                project_instructions=project_instructions,
                available_skills=available_skills,
                active_skills=active_skills,
            )
            if body and body.strip()
        }

    def _sections(
        self,
        *,
        config: SessionConfig,
        context: TurnContext,
        project_instructions: str | None,
        available_skills: str | None,
        active_skills: str | None,
    ) -> list[tuple[str, str | None]]:
        """返回从稳定规则到 Turn 快照的固定 section 顺序。"""
        return [
            ("base_instructions", config.base_instructions or BASE_INSTRUCTIONS),
            ("project_instructions", project_instructions),
            ("user_instructions", config.user_instructions),
            ("available_skills", available_skills),
            ("active_skills", active_skills),
            ("turn_context", self._turn_context(context)),
        ]

    @staticmethod
    def _render_section(name: str, body: str) -> str:
        """去除正文首尾空白并包裹成边界清晰的 XML 风格 section。

        Example:
            >>> PromptBuilder._render_section("rules", "  be safe  ")
            '<rules>\\nbe safe\\n</rules>'
        """
        return f"<{name}>\n{body.strip()}\n</{name}>"

    @staticmethod
    def _turn_context(context: TurnContext) -> str:
        """只向模型暴露当前 Turn 需要理解的执行与安全状态。"""
        return "\n".join(
            [
                f"cwd: {context.cwd}",
                f"approval_policy: {context.approval_policy}",
                f"sandbox_mode: {context.sandbox_mode}",
                f"network_access: {str(context.network_access).lower()}",
            ]
        )
