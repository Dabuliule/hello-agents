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
    resolved = path.expanduser().resolve(strict=False)
    root = workspace_root.expanduser().resolve(strict=False)
    return resolved == root or root in resolved.parents


def display_path(path: Path, workspace_root: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    root = workspace_root.expanduser().resolve(strict=False)
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(path)


def looks_binary(raw: bytes) -> bool:
    return b"\0" in raw[:4096]
