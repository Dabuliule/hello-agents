from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from codecraft.skill import SkillMetadata


class ComposerChoiceKind(StrEnum):
    """Composer 候选是本地命令还是待插入的 Skill mention。"""

    COMMAND = "command"
    SKILL = "skill"


class ComposerMenuMode(StrEnum):
    """菜单当前搜索 slash commands 或 Skills。"""

    COMMANDS = "commands"
    SKILLS = "skills"


@dataclass(frozen=True)
class SlashCommand:
    """本地 slash command 的稳定名称和用户描述。"""

    name: str
    description: str


@dataclass(frozen=True)
class ComposerChoice:
    """OptionList 使用的唯一 ID、行为类型、值和展示文本。"""

    id: str
    kind: ComposerChoiceKind
    value: str
    title: str
    description: str


@dataclass(frozen=True)
class ComposerMenuQuery:
    """解析出的菜单模式、查询词和选中 Skill 要替换的字符区间。"""

    mode: ComposerMenuMode
    query: str
    replace_start: int
    replace_end: int


SLASH_COMMANDS = (
    SlashCommand("skills", "Browse and use local skills"),
    SlashCommand("status", "Show session and runtime status"),
    SlashCommand("tools", "List tools available to the model"),
    SlashCommand("mcp", "List configured MCP servers"),
    SlashCommand("trace", "Inspect the current session trace"),
    SlashCommand("quit", "Exit CodeCraft"),
)


def parse_composer_menu(value: str) -> ComposerMenuQuery | None:
    """识别 slash command 或最后一个 `$skill` token 的补全请求。"""
    if value == "/skills" or value.startswith("/skills "):
        return ComposerMenuQuery(
            mode=ComposerMenuMode.SKILLS,
            query=value[len("/skills") :].strip(),
            replace_start=0,
            replace_end=len(value),
        )

    if value.startswith("/") and " " not in value:
        return ComposerMenuQuery(
            mode=ComposerMenuMode.COMMANDS,
            query=value[1:],
            replace_start=0,
            replace_end=len(value),
        )

    token_start = value.rfind(" ") + 1
    token = value[token_start:]
    if token.startswith("$"):
        return ComposerMenuQuery(
            mode=ComposerMenuMode.SKILLS,
            query=token[1:],
            replace_start=token_start,
            replace_end=len(value),
        )
    return None


def command_choices(query: str) -> tuple[ComposerChoice, ...]:
    """按名称或描述的大小写不敏感子串过滤 slash commands。"""
    normalized = query.casefold()
    return tuple(
        ComposerChoice(
            id=f"command-{command.name}",
            kind=ComposerChoiceKind.COMMAND,
            value=command.name,
            title=f"/{command.name}",
            description=command.description,
        )
        for command in SLASH_COMMANDS
        if not normalized
        or normalized in command.name.casefold()
        or normalized in command.description.casefold()
    )


def skill_choices(
    skills: tuple[SkillMetadata, ...],
    query: str,
) -> tuple[ComposerChoice, ...]:
    """按 Skill 名/描述过滤，并在候选描述显示 USER/PROJECT 来源。"""
    normalized = query.casefold()
    return tuple(
        ComposerChoice(
            id=f"skill-{metadata.name}",
            kind=ComposerChoiceKind.SKILL,
            value=metadata.name,
            title=f"${metadata.name}",
            description=f"{metadata.description}  [{metadata.source.value}]",
        )
        for metadata in skills
        if not normalized
        or normalized in metadata.name.casefold()
        or normalized in metadata.description.casefold()
    )
