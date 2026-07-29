from __future__ import annotations

from pathlib import Path

from codecraft.core.errors import WorkspaceAccessError


class WorkspaceGuard:
    """把用户传入的路径限制在当前 session 的工作目录内。

    所有文件读写工具都应先经过这个 guard。它允许不存在的目标路径用于写入，
    但解析后的最终路径仍必须落在 workspace 内。
    """

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()

    def resolve_read_path(self, path: str) -> Path:
        """解析读取路径，并确认它没有逃出 workspace。"""
        resolved = self._resolve(path)
        self.assert_inside_workspace(resolved)
        return resolved

    def resolve_write_path(self, path: str) -> Path:
        """解析写入路径，并确认它没有逃出 workspace。"""
        resolved = self._resolve(path)
        self.assert_inside_workspace(resolved)
        return resolved

    def assert_inside_workspace(self, path: Path) -> None:
        """检查路径是否在当前 workspace 下。"""
        resolved = path.expanduser().resolve(strict=False)
        if resolved == self.workspace or self.workspace in resolved.parents:
            return

        raise WorkspaceAccessError(
            "path is outside workspace",
            code="workspace_access_denied",
            suggestion="Use a path inside the current workspace.",
            metadata={"path": str(path)},
        )

    def _resolve(self, path: str) -> Path:
        if not path:
            raise WorkspaceAccessError(
                "path must not be empty",
                code="workspace_path_empty",
            )

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)
