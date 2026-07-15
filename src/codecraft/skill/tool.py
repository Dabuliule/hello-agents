from __future__ import annotations

from pydantic import BaseModel, Field

from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.skill.models import SKILL_NAME_PATTERN
from codecraft.skill.registry import SkillRegistry
from codecraft.tool.base import BaseTool, ToolArguments, ToolContext


SKILL_ACTIVATION_METADATA_KEY = "skill_activation"


class LoadSkillArgs(ToolArguments):
    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=SKILL_NAME_PATTERN.pattern,
    )


class LoadSkillTool(BaseTool):
    """让模型按需激活一个已发现的 Skill，不直接返回 Skill 正文。"""

    name = "load_skill"
    description = (
        "Activate one available skill for the current turn. Use the exact skill name "
        "from <available_skills>."
    )
    args_schema = LoadSkillArgs
    effects = {ToolEffect.READ_ONLY}

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        load_args = LoadSkillArgs.model_validate(args)
        skill = self.registry.get(load_args.name)
        metadata = skill.metadata
        return ToolResult(
            success=True,
            content=(
                f"Skill '{metadata.name}' is active for the current turn. "
                "Follow its instructions from <active_skills> in the next model request."
            ),
            data={
                "name": metadata.name,
                "source": metadata.source.value,
                "path": str(metadata.path),
            },
            metadata={SKILL_ACTIVATION_METADATA_KEY: metadata.name},
        )
