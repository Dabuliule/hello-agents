from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError
import yaml
from yaml import YAMLError

from codecraft.core.errors import CodecraftError
from codecraft.core.token_budget import estimate_text_tokens
from codecraft.skill.models import (
    Skill,
    SkillDiagnostic,
    SkillManifest,
    SkillMetadata,
    SkillSource,
)


class SkillNotFoundError(CodecraftError):
    pass


class SkillRegistry:
    """发现、校验并保存当前 runtime 可用的 Skills。"""

    DEFAULT_MAX_FILE_BYTES = 64 * 1024
    _EXPLICIT_MENTION = re.compile(
        r"(?<![A-Za-z0-9_$])\$([a-z0-9][a-z0-9_-]{0,63})(?![A-Za-z0-9_-])"
    )

    def __init__(
        self,
        skills: Iterable[Skill] = (),
        diagnostics: Iterable[SkillDiagnostic] = (),
    ) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            name = skill.metadata.name
            if name in self._skills:
                raise ValueError(f"skill already registered: {name}")
            self._skills[name] = skill
        self._diagnostics = tuple(diagnostics)

    @classmethod
    def discover(
        cls,
        *,
        user_root: Path,
        project_root: Path,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> SkillRegistry:
        """扫描两级 Skill 目录；无效条目记为诊断，不阻断应用启动。"""
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")

        # 这里只消除相对路径，不跟随 symlink；两类根目录必须分别接受安全校验。
        normalized_user = user_root.expanduser().absolute()
        normalized_project = project_root.expanduser().absolute()
        roots = (
            [(SkillSource.PROJECT, project_root)]
            if normalized_user == normalized_project
            else [
                (SkillSource.USER, user_root),
                (SkillSource.PROJECT, project_root),
            ]
        )

        selected: dict[str, Skill] = {}
        diagnostics: list[SkillDiagnostic] = []
        for source, root in roots:
            discovered, root_diagnostics = cls._discover_root(
                root,
                source=source,
                max_file_bytes=max_file_bytes,
            )
            diagnostics.extend(root_diagnostics)
            for skill in discovered:
                previous = selected.get(skill.metadata.name)
                if previous is not None:
                    diagnostics.append(
                        SkillDiagnostic(
                            code="skill_shadowed",
                            message=(
                                f"{previous.metadata.source.value} skill "
                                f"'{skill.metadata.name}' is shadowed by the project skill"
                            ),
                            path=previous.metadata.path,
                            source=previous.metadata.source,
                        )
                    )
                selected[skill.metadata.name] = skill

        return cls(
            (selected[name] for name in sorted(selected)),
            diagnostics,
        )

    @classmethod
    def _discover_root(
        cls,
        root: Path,
        *,
        source: SkillSource,
        max_file_bytes: int,
    ) -> tuple[list[Skill], list[SkillDiagnostic]]:
        root = root.expanduser()
        if not root.exists():
            return [], []
        if root.is_symlink():
            return [], [
                cls._diagnostic(
                    "skill_root_symlink",
                    "skill root must not be a symbolic link",
                    root,
                    source,
                )
            ]
        if not root.is_dir():
            return [], [
                cls._diagnostic(
                    "invalid_skill_root",
                    "skill root must be a directory",
                    root,
                    source,
                )
            ]

        skills: list[Skill] = []
        diagnostics: list[SkillDiagnostic] = []
        try:
            entries = sorted(root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            return [], [
                cls._diagnostic(
                    "skill_root_unreadable",
                    f"skill root could not be read: {exc}",
                    root,
                    source,
                )
            ]

        for directory in entries:
            if directory.is_symlink():
                diagnostics.append(
                    cls._diagnostic(
                        "skill_directory_symlink",
                        "skill directory must not be a symbolic link",
                        directory,
                        source,
                    )
                )
                continue
            if not directory.is_dir():
                continue

            skill_path = directory / "SKILL.md"
            if not skill_path.exists():
                continue
            try:
                skill = cls._load_skill(
                    skill_path,
                    directory_name=directory.name,
                    source=source,
                    max_file_bytes=max_file_bytes,
                )
            except (
                OSError,
                UnicodeError,
                ValueError,
                ValidationError,
                YAMLError,
            ) as exc:
                diagnostics.append(
                    cls._diagnostic(
                        "invalid_skill",
                        str(exc),
                        skill_path,
                        source,
                    )
                )
                continue
            skills.append(skill)

        return skills, diagnostics

    @staticmethod
    def _load_skill(
        path: Path,
        *,
        directory_name: str,
        source: SkillSource,
        max_file_bytes: int,
    ) -> Skill:
        if path.is_symlink():
            raise ValueError("SKILL.md must not be a symbolic link")
        if not path.is_file():
            raise ValueError("SKILL.md must be a regular file")
        if path.stat().st_size > max_file_bytes:
            raise ValueError(f"SKILL.md exceeds {max_file_bytes} bytes")

        raw = path.read_bytes()
        if len(raw) > max_file_bytes:
            raise ValueError(f"SKILL.md exceeds {max_file_bytes} bytes")
        text = raw.decode("utf-8")
        frontmatter, instructions = SkillRegistry._split_document(text)
        parsed: Any = yaml.safe_load(frontmatter)
        if not isinstance(parsed, dict):
            raise ValueError("skill frontmatter must be a YAML mapping")
        manifest = SkillManifest.model_validate(parsed)
        if manifest.name != directory_name:
            raise ValueError("skill name must match its directory name")

        return Skill(
            metadata=SkillMetadata(
                name=manifest.name,
                description=manifest.description,
                source=source,
                path=path.resolve(),
            ),
            instructions=instructions,
        )

    @staticmethod
    def _split_document(text: str) -> tuple[str, str]:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            raise ValueError("SKILL.md must start with YAML frontmatter")

        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is None:
            raise ValueError("SKILL.md frontmatter is not closed")

        frontmatter = "".join(lines[1:closing_index])
        instructions = "".join(lines[closing_index + 1 :]).strip()
        if not instructions:
            raise ValueError("SKILL.md instructions must not be blank")
        return frontmatter, instructions

    @staticmethod
    def _diagnostic(
        code: str,
        message: str,
        path: Path,
        source: SkillSource,
    ) -> SkillDiagnostic:
        return SkillDiagnostic(
            code=code,
            message=message,
            path=path.resolve(strict=False),
            source=source,
        )

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = sorted(self._skills)
            raise SkillNotFoundError(
                f"skill not found: {name}",
                code="skill_not_found",
                suggestion=(
                    f"Choose one of: {', '.join(available)}"
                    if available
                    else "No valid skills were discovered."
                ),
                metadata={"skill": name, "available_skills": available},
            ) from exc

    def list(self) -> tuple[SkillMetadata, ...]:
        return tuple(skill.metadata for skill in self._skills.values())

    def diagnostics(self) -> tuple[SkillDiagnostic, ...]:
        return self._diagnostics

    def explicit_mentions(self, text: str) -> tuple[Skill, ...]:
        """按出现顺序解析有效的 `$skill-name`，重复 mention 只保留一次。"""
        names = dict.fromkeys(
            match.group(1) for match in self._EXPLICIT_MENTION.finditer(text)
        )
        return tuple(self._skills[name] for name in names if name in self._skills)

    def __bool__(self) -> bool:
        return bool(self._skills)

    def catalogue_prompt(self, *, max_tokens: int | None = None) -> str | None:
        """返回确定性的 Skill 目录；超出预算时只移除完整条目。"""
        if not self._skills:
            return None
        catalogue = [
            {
                "name": metadata.name,
                "description": metadata.description,
                "source": metadata.source.value,
            }
            for metadata in self.list()
        ]
        prompt = self._prompt_json(catalogue)
        if max_tokens is None or estimate_text_tokens(prompt) <= max_tokens:
            return prompt
        if max_tokens <= 0:
            return None

        selected: list[dict[str, str]] = []
        for entry in catalogue:
            candidate_entries = [*selected, entry]
            candidate = self._prompt_json(
                {
                    "skills": candidate_entries,
                    "omitted_count": len(catalogue) - len(candidate_entries),
                }
            )
            if estimate_text_tokens(candidate) > max_tokens:
                break
            selected = candidate_entries

        bounded = self._prompt_json(
            {
                "skills": selected,
                "omitted_count": len(catalogue) - len(selected),
            }
        )
        return bounded if estimate_text_tokens(bounded) <= max_tokens else None

    @staticmethod
    def _prompt_json(value: Any) -> str:
        return (
            json.dumps(value, ensure_ascii=False, indent=2)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
        )

    @staticmethod
    def active_prompt(skills: Iterable[Skill]) -> str | None:
        sections = [
            "\n".join(
                [
                    f"## Skill: {skill.metadata.name}",
                    f"Source: {skill.metadata.source.value}",
                    f"Path: {skill.metadata.path}",
                    "",
                    skill.instructions,
                ]
            )
            for skill in skills
        ]
        return "\n\n".join(sections) or None
