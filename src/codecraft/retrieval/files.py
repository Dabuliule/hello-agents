from __future__ import annotations

from pathlib import Path

SKIPPED_NAMES = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".uv-cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)


def iter_workspace_files(root: Path) -> list[Path]:
    """确定性列出根内普通文件，跳过缓存、构建产物和逃逸 symlink。

    Args:
        root: 要遍历的 workspace 根目录。

    Returns:
        按绝对路径排序的安全文件列表。
    """
    workspace_root = root.expanduser().resolve()
    return sorted(
        path
        for path in workspace_root.rglob("*")
        if path.is_file()
        and is_inside_workspace(path, workspace_root)
        and not any(
            part in SKIPPED_NAMES for part in path.relative_to(workspace_root).parts
        )
    )


def is_inside_workspace(path: Path, workspace_root: Path) -> bool:
    """按 resolve 后路径判断目标是否等于 workspace 或位于其下。"""
    resolved = path.expanduser().resolve(strict=False)
    root = workspace_root.expanduser().resolve(strict=False)
    return resolved == root or root in resolved.parents


def display_path(path: Path, workspace_root: Path) -> str:
    """workspace 内返回相对展示路径，外部路径保留调用方原表示。"""
    resolved = path.expanduser().resolve(strict=False)
    root = workspace_root.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(path)


def looks_binary(raw: bytes) -> bool:
    """用前 4 KiB 是否含 NUL 字节快速识别不应文本检索的内容。

    Example:
        >>> looks_binary(b"hello"), looks_binary(b"a\\x00b")
        (False, True)
    """
    return b"\0" in raw[:4096]
