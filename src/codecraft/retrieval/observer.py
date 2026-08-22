from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from codecraft.core.turn_context import TurnContext
from codecraft.retrieval.errors import RetrievalUnavailableError
from codecraft.retrieval.files import is_inside_workspace
from codecraft.retrieval.index import RepositoryIndex
from codecraft.schema.tool import ToolCall, ToolResult


class WorkspaceIndexObserver:
    """在成功 write_file/apply_patch 后增量刷新已存在的仓库索引。

    Observer 只消费工具返回的结构化实际变更路径，不根据调用参数猜测副作用；索引未
    预先构建或后处理失败也不能反向判定原文件写入失败。Bash 等通用进程工具没有可靠
    changed_files 协议，因此不在此自动刷新，其外部修改由查询新鲜度检查和 Scan 降级
    尽力兜底。
    """

    name = "workspace_index"

    def __init__(self, index: RepositoryIndex) -> None:
        """绑定负责所有 workspace 数据库的 RepositoryIndex。"""
        self.index = index

    async def after_result(
        self,
        call: ToolCall,
        result: ToolResult,
        context: TurnContext,
    ) -> dict[str, Any] | None:
        """从工具结果提取实际变更文件并在线程中刷新索引。

        Returns:
            非写操作/失败/无变更返回 ``None``；成功刷新返回计数。索引尚未
            构建时返回 ``status=skipped``，不会让原工具调用失败。
        """
        paths = _changed_paths(call, result)
        if not paths:
            return None

        root = context.cwd.expanduser().resolve()
        changed = _paths_in_workspace(paths, root)
        if not changed:
            return None

        try:
            stats = await asyncio.to_thread(
                self.index.refresh_paths,
                root,
                changed,
            )
        except RetrievalUnavailableError:
            update: dict[str, Any] = {
                "workspace": str(root),
                "status": "skipped",
                "reason": "index_not_built",
            }
        else:
            update = {
                "workspace": str(root),
                "status": "updated",
                "updated_files": stats.updated_file_count,
                "unchanged_files": stats.unchanged_file_count,
                "deleted_files": stats.deleted_file_count,
                "indexed_bytes": stats.indexed_bytes,
            }
        return update


def _changed_paths(call: ToolCall, result: ToolResult) -> list[Path]:
    """只信任成功 write_file/apply_patch 的结构化结果提取变更路径。"""
    if not result.success or result.data is None:
        return []
    if call.name == "write_file":
        if result.data.get("changed") is not True:
            return []
        path = result.data.get("path")
        return [Path(path)] if isinstance(path, str) else []
    if call.name == "apply_patch":
        paths = result.data.get("changed_files")
        if not isinstance(paths, list):
            return []
        return [Path(path) for path in paths if isinstance(path, str)]
    return []


def _paths_in_workspace(paths: list[Path], workspace_root: Path) -> list[Path]:
    """解析并保留 workspace 内路径，阻止观察器刷新外部文件。"""
    selected: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve(strict=False)
        if is_inside_workspace(resolved, workspace_root):
            selected.append(resolved)
    return selected
