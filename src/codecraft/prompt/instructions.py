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
        target_paths: Iterable[Path] = (),
    ) -> str | None:
        """安全读取并合并 workspace 根与目标路径作用域内的项目规则。

        Args:
            cwd: 当前 workspace 根，也是允许读取 instructions 的安全边界。
            target_paths: 当前 Turn 已访问或准备访问的相对/绝对文件路径。

        Returns:
            带来源与 scope 标题、由根到深层排列的合并文本；没有规则时返回
            ``None``，超限时优先保留较深、较新的作用域。
        """
        current = cwd.expanduser().resolve()

        directories = _instruction_directories(
            current=current,
            target_paths=target_paths,
        )
        sections = _load_instruction_sections(
            directories=directories,
            workspace=current,
            filenames=self.filenames,
            max_chars=self.max_chars,
        )

        if not sections:
            return None

        return _bounded_sections(sections, max_chars=self.max_chars)


def _instruction_directories(
    *,
    current: Path,
    target_paths: Iterable[Path],
) -> list[Path]:
    """构造根目录和每个安全 target 从根到叶子的规则搜索顺序。"""
    directories = [current]
    for target in target_paths:
        resolved = _resolve_target(target, cwd=current)
        if resolved is None:
            continue
        target_directory = resolved if resolved.is_dir() else resolved.parent
        directories.extend(reversed(_walk_up(target_directory, current)))
    return directories


def _load_instruction_sections(
    *,
    directories: list[Path],
    workspace: Path,
    filenames: tuple[str, ...],
    max_chars: int,
) -> list[str]:
    """安全读取未重复的规则文件，并附加相对来源和作用域标题。"""
    sections: list[str] = []
    seen: set[Path] = set()
    for directory in directories:
        if not _is_inside_workspace(directory, workspace):
            continue
        for filename in filenames:
            path = directory / filename
            safe_path = _safe_instruction_path(path, workspace)
            if safe_path is None or safe_path in seen:
                continue
            seen.add(safe_path)
            content = _read_text(safe_path, max_chars=max_chars)
            if content is None or not content.strip():
                continue
            source = path.relative_to(workspace)
            scope = path.parent.relative_to(workspace)
            scope_label = str(scope) if scope.parts else "."
            sections.append(f"# {source} (scope: {scope_label})\n\n{content.strip()}")
    return sections


def _walk_up(start: Path, stop: Path) -> list[Path]:
    """返回从 start 到祖先 stop 的目录链，包含两端。

    Raises:
        ValueError: ``stop`` 不是 ``start`` 本身或祖先。
    """
    if start != stop and stop not in start.parents:
        raise ValueError("stop must contain start")
    directories = [start]
    current = start
    while current != stop:
        current = current.parent
        directories.append(current)
    return directories


def _resolve_target(path: Path, *, cwd: Path) -> Path | None:
    """相对 cwd 解析 target；解析结果逃出 workspace 时返回 ``None``。"""
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve(strict=False)
    return resolved if _is_inside_workspace(resolved, cwd) else None


def _is_inside_workspace(path: Path, workspace: Path) -> bool:
    """判断路径是 workspace 本身或其后代。"""
    return path == workspace or workspace in path.parents


def _safe_instruction_path(path: Path, root: Path) -> Path | None:
    """解析真实文件并拒绝不存在、非文件或 symlink 逃逸 workspace 的路径。"""
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved if resolved == root or root in resolved.parents else None


def _read_text(path: Path, *, max_chars: int) -> str | None:
    """最多读取 ``max_chars + 1`` 字符，I/O 或 UTF-8 失败时安全跳过。"""
    try:
        with path.open("r", encoding="utf-8") as stream:
            return stream.read(max_chars + 1)
    except (OSError, UnicodeDecodeError):
        return None


def _bounded_sections(sections: list[str], *, max_chars: int) -> str:
    """在字符上限内优先保留靠后的深层规则和完整 section。

    Example:
        >>> _bounded_sections(["root", "deep"], max_chars=100).splitlines()
        ['root', '', 'deep']
    """
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
