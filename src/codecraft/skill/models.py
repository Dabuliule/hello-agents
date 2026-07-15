from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class SkillSource(StrEnum):
    USER = "user"
    PROJECT = "project"


class SkillManifest(BaseModel):
    """`SKILL.md` YAML frontmatter 中受支持的字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=500)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not SKILL_NAME_PATTERN.fullmatch(normalized):
            raise ValueError(
                "name must use lowercase letters, digits, hyphens, or underscores"
            )
        return normalized

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("description must not be blank")
        if "\n" in normalized or "\r" in normalized:
            raise ValueError("description must be a single line")
        if any(ord(char) < 32 and char != "\t" for char in normalized):
            raise ValueError("description must not contain control characters")
        return normalized


class SkillMetadata(BaseModel):
    """可提前暴露给模型的轻量 Skill 摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    source: SkillSource
    path: Path


class Skill(BaseModel):
    """通过校验并已读入内存的 Skill。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: SkillMetadata
    instructions: str = Field(min_length=1)


class SkillDiagnostic(BaseModel):
    """发现阶段的非致命问题，供诊断和后续 UI 展示。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    path: Path
    source: SkillSource
