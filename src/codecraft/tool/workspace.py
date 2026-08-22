from __future__ import annotations

from pathlib import Path

from codecraft.core.errors import WorkspaceAccessError


class WorkspaceGuard:
    """把工具参数中的路径限制在当前 Session 的真实 workspace 边界内。

    Guard 先展开 ``~``、锚定相对路径并解析 ``..`` 与已有 symlink，再用 Path 的
    parent 关系判断归属，避免字符串前缀把 ``/repo-evil`` 误认成 ``/repo``。
    ``strict=False`` 允许写入目标尚不存在，但其已存在的父路径和最终规范路径仍
    必须位于 workspace。

    这是文件工具的应用层路径校验，不是 OS 级隔离：任意命令仍需 SandboxBackend
    约束；面对不可信本机进程并发替换目录的 TOCTOU，还需要 fd/openat 等更强机制。
    """

    def __init__(self, workspace: Path) -> None:
        """将已验证的 Session cwd 规范化为后续工具共享的真实绝对边界。"""
        self.workspace = workspace.expanduser().resolve()

    def resolve_read_path(self, path: str) -> Path:
        """返回 workspace 内的规范读取路径；存在性和文件类型由具体工具检查。"""
        resolved = self._resolve(path)
        self.assert_inside_workspace(resolved)
        return resolved

    def resolve_write_path(self, path: str) -> Path:
        """返回 workspace 内的规范写入路径，并允许最终目标暂时不存在。

        read/write 当前共享相同的归属算法，但保留两个入口表达调用意图，也为未来
        单独增加只读挂载、禁止覆盖或父目录策略留下稳定 API 边界。
        """
        resolved = self._resolve(path)
        self.assert_inside_workspace(resolved)
        return resolved

    def assert_inside_workspace(self, path: Path) -> None:
        """按规范 Path 的祖先关系检查归属，而不是使用不安全的字符串前缀。"""
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
        """把相对、绝对或用户目录路径统一成可做归属判断的真实绝对路径。

        Raises:
            WorkspaceAccessError: path 为空；越界由公开 resolve 方法随后检查。

        相对路径只能以 workspace 为锚；绝对路径和 ``~`` 不会自动拒绝，而是在
        canonicalize 后接受同一套归属判断。解析已有 symlink 可阻止 workspace 内
        链接指向外部文件，``strict=False`` 则保留创建新文件的能力。
        """
        if not path:
            raise WorkspaceAccessError(
                "path must not be empty",
                code="workspace_path_empty",
            )

        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)
