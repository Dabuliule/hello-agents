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
    name = "lexical"

    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
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
    name = "symbol"

    def __init__(self, index: RepositoryIndex) -> None:
        self.index = index

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
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
    resolved = request.root.resolve(strict=False)
    workspace_root = request.workspace_root.resolve(strict=False)
    try:
        relative = resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise RetrievalUnavailableError(
            "request root is outside indexed workspace"
        ) from exc
    return workspace_root, str(relative) if relative.parts else "."
