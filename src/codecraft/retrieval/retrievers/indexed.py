from __future__ import annotations

import asyncio
from pathlib import Path

from codecraft.retrieval.errors import RetrievalUnavailableError
from codecraft.retrieval.index import RepositoryIndex
from codecraft.retrieval.models import (
    MatchType,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStats,
)
from codecraft.retrieval.retrievers.base import Retriever


class LexicalRetriever(Retriever):
    """把 RepositoryIndex 的 FTS/path 查询适配为标准 RetrievalResponse。"""

    name = "lexical"

    def __init__(self, index: RepositoryIndex) -> None:
        """绑定一个可按 workspace 定位数据库的仓库索引。"""
        self.index = index

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """在线程池查询 SQLite，并在任何命中已陈旧时请求上层降级。"""
        workspace_root, scope = _workspace_and_scope(request)
        result = await asyncio.to_thread(
            self.index.search_lexical,
            workspace_root,
            query=request.query,
            scope=scope,
            mode=request.mode,
            case_sensitive=request.case_sensitive,
            max_results=request.max_results,
        )
        if result.stale_file_count:
            raise RetrievalUnavailableError("indexed matches include stale files")
        match_type: MatchType = "path" if request.mode == "path" else "content"
        return RetrievalResponse(
            matches=tuple(
                RetrievalMatch(
                    type=match_type,
                    path=match.path,
                    line=None if match_type == "path" else match.line,
                    snippet=None if match_type == "path" else match.snippet,
                )
                for match in result.matches
            ),
            stats=RetrievalStats(
                candidate_file_count=result.indexed_file_count,
                skipped={"stale": result.stale_file_count},
            ),
            truncated=result.truncated,
        )


class SymbolRetriever(Retriever):
    """按 Tree-sitter 抽取的符号精确名或前缀检索代码定义。"""

    name = "symbol"

    def __init__(self, index: RepositoryIndex) -> None:
        """绑定 RepositoryIndex。"""
        self.index = index

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """检索符号并拒绝 path 模式或包含陈旧文件的结果。"""
        if request.mode == "path":
            raise RetrievalUnavailableError("symbol retrieval does not search paths")
        workspace_root, scope = _workspace_and_scope(request)
        result = await asyncio.to_thread(
            self.index.search_symbols,
            workspace_root,
            query=request.query,
            scope=scope,
            case_sensitive=request.case_sensitive,
            max_results=request.max_results,
        )
        if result.stale_file_count:
            raise RetrievalUnavailableError("indexed symbols include stale files")
        return RetrievalResponse(
            matches=tuple(
                RetrievalMatch(
                    type="content",
                    path=match.path,
                    line=match.line,
                    snippet=match.snippet,
                )
                for match in result.matches
            ),
            stats=RetrievalStats(
                candidate_file_count=result.indexed_file_count,
                skipped={"stale": result.stale_file_count},
            ),
            truncated=result.truncated,
        )


def _workspace_and_scope(request: RetrievalRequest) -> tuple[Path, str]:
    """把 request.root 安全转换成索引 workspace 下的相对 scope。

    Raises:
        RetrievalUnavailableError: root 不位于索引 workspace 内。
    """
    resolved = request.root.resolve(strict=False)
    workspace_root = request.workspace_root.resolve(strict=False)
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise RetrievalUnavailableError(
            "request root is outside indexed workspace"
        ) from exc
    return workspace_root, str(relative) if relative.parts else "."
