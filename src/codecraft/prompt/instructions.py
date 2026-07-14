from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstructionLoader:
    """从 workspace 中收集项目级 instructions。

    根目录规则先出现，越深的规则越靠后并覆盖上层规则。target_paths 用于
    补充当前 turn 已访问文件对应的目录作用域。
    """

    filenames: tuple[str, ...] = ("AGENTS.md", "CODECRAFT.md")
    max_chars: int = 40_000

    def load_project_instructions(
        self,
        *,
        cwd: Path,
        workspace_roots: list[Path],
        target_paths: Iterable[Path] = (),
    ) -> str | None:
        """安全读取并合并当前目录及目标路径可见的项目 instructions。"""
        roots = [root.expanduser().resolve() for root in workspace_roots]
        current = cwd.expanduser().resolve()
        matched_root = _find_containing_root(current, roots)
        if matched_root is None:
            return None

        directories = list(reversed(_walk_up(current, matched_root)))
        for target in target_paths:
            resolved = _resolve_target(target, cwd=current, roots=roots)
            if resolved is None:
                continue
            target_root = _find_containing_root(resolved, roots)
            if target_root is None:
                continue
            target_directory = resolved if resolved.is_dir() else resolved.parent
            directories.extend(reversed(_walk_up(target_directory, target_root)))

        sections: list[str] = []
        seen: set[Path] = set()
        for directory in directories:
            directory_root = _find_containing_root(directory, roots)
            if directory_root is None:
                continue
            for filename in self.filenames:
                path = directory / filename
                safe_path = _safe_instruction_path(path, directory_root)
                if safe_path is None or safe_path in seen:
                    continue
                seen.add(safe_path)
                content = _read_text(safe_path, max_chars=self.max_chars)
                if content is None or not content.strip():
                    continue
                source = path.relative_to(directory_root)
                scope = path.parent.relative_to(directory_root)
                scope_label = str(scope) if scope.parts else "."
                sections.append(
                    f"# {source} (scope: {scope_label})\n\n{content.strip()}"
                )

        if not sections:
            return None

        return _bounded_sections(sections, max_chars=self.max_chars)


def _find_containing_root(path: Path, roots: list[Path]) -> Path | None:
    """找到包含 path 的最深 workspace root。"""
    containing = [root for root in roots if path == root or root in path.parents]
    if not containing:
        return None
    return max(containing, key=lambda root: len(root.parts))


def _walk_up(start: Path, stop: Path) -> list[Path]:
    """返回从 start 到 stop 的目录链，包含两端。"""
    directories = [start]
    current = start
    while current != stop:
        current = current.parent
        directories.append(current)
    return directories


def _resolve_target(path: Path, *, cwd: Path, roots: list[Path]) -> Path | None:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve(strict=False)
    return resolved if _find_containing_root(resolved, roots) is not None else None


def _safe_instruction_path(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved if resolved == root or root in resolved.parents else None


def _read_text(path: Path, *, max_chars: int) -> str | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return stream.read(max_chars + 1)
    except (OSError, UnicodeDecodeError):
        return None


def _bounded_sections(sections: list[str], *, max_chars: int) -> str:
    combined = "\n\n".join(sections)
    if len(combined) <= max_chars:
        return combined

    marker = "[earlier project instructions omitted]\n\n"
    if max_chars <= len(marker):
        return marker[:max_chars]
    remaining = max(0, max_chars - len(marker))
    selected: list[str] = []
    for section in reversed(sections):
        separator_chars = 2 if selected else 0
        if len(section) + separator_chars <= remaining:
            selected.append(section)
            remaining -= len(section) + separator_chars
            continue
        if not selected and remaining > 0:
            selected.append(section[:remaining].rstrip())
        break
    return marker + "\n\n".join(reversed(selected))
