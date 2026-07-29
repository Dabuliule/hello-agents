from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from mcp import types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from codecraft.prompt import InstructionLoader
from codecraft.retrieval import (
    ContextEngine,
    LexicalRetriever,
    RepositoryIndex,
    RetrievalRequest,
    ScanRetriever,
    SymbolRetriever,
)
from codecraft.tool import WorkspaceGuard


class RepositoryMatchResult(BaseModel):
    type: Literal["content", "path"]
    path: str
    line: int | None = None
    snippet: str | None = None


class RepositorySearchResult(BaseModel):
    query: str
    path: str
    matches: list[RepositoryMatchResult]
    match_count: int
    truncated: bool
    retriever: str | None
    fallback_from: str | None
    route_reason: str | None
    attempted_retrievers: list[str]
    candidate_file_count: int
    scanned_file_count: int
    read_file_count: int
    scanned_bytes: int
    skipped: dict[str, int]


def create_repository_mcp_server(
    workspace: Path,
    *,
    codecraft_home: Path | None = None,
) -> FastMCP:
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"workspace must be a directory: {root}")
    home = (codecraft_home or Path("~/.codecraft")).expanduser().resolve()
    index = RepositoryIndex(home / "indexes")
    engine = ContextEngine(
        [
            ScanRetriever(),
            LexicalRetriever(index),
            SymbolRetriever(index),
        ]
    )
    guard = WorkspaceGuard(root)
    server = FastMCP(
        "CodeCraft Repository Context",
        instructions=(
            "Use search_repository to locate paths, text, and symbols inside the "
            "configured repository. This server is read-only."
        ),
        log_level="WARNING",
    )

    @server.tool(
        name="search_repository",
        description=(
            "Search the configured repository for paths, text, or indexed symbols."
        ),
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
        structured_output=True,
    )
    async def search_repository(
        query: Annotated[str, Field(min_length=1)],
        path: str = ".",
        mode: Literal["both", "content", "path"] = "both",
        strategy: Literal["auto", "scan", "lexical", "symbol"] = "auto",
        case_sensitive: bool = False,
        max_results: Annotated[int, Field(ge=1, le=100)] = 20,
        max_file_bytes: Annotated[int, Field(ge=1, le=10_000_000)] = 1_000_000,
    ) -> RepositorySearchResult:
        search_root = guard.resolve_read_path(path)
        if not search_root.is_dir():
            raise ValueError(f"search path must be a directory: {path}")
        response = await engine.retrieve(
            RetrievalRequest(
                query=query,
                root=search_root,
                workspace_root=root,
                mode=mode,
                case_sensitive=case_sensitive,
                max_results=max_results,
                max_file_bytes=max_file_bytes,
            ),
            retriever_name=strategy,
            fallback_retriever="scan",
        )
        stats = response.stats
        return RepositorySearchResult(
            query=query,
            path=search_root.relative_to(root).as_posix(),
            matches=[
                RepositoryMatchResult(**match.as_dict()) for match in response.matches
            ],
            match_count=len(response.matches),
            truncated=response.truncated,
            retriever=response.retriever,
            fallback_from=response.fallback_from,
            route_reason=response.route_reason,
            attempted_retrievers=list(response.attempted_retrievers),
            candidate_file_count=stats.candidate_file_count,
            scanned_file_count=stats.scanned_file_count,
            read_file_count=stats.read_file_count,
            scanned_bytes=stats.scanned_bytes,
            skipped=stats.skipped,
        )

    @server.resource(
        "codecraft://workspace/metadata",
        name="workspace_metadata",
        description="Metadata for the repository served by CodeCraft.",
        mime_type="application/json",
    )
    def workspace_metadata() -> str:
        return json.dumps(
            {
                "workspace": str(root),
                "index_available": index.database_path(root).is_file(),
                "retrievers": list(engine.retriever_names),
            },
            sort_keys=True,
        )

    @server.resource(
        "codecraft://workspace/instructions",
        name="workspace_instructions",
        description="AGENTS.md and CODECRAFT.md instructions visible at workspace root.",
        mime_type="text/markdown",
    )
    def workspace_instructions() -> str:
        return (
            InstructionLoader().load_project_instructions(cwd=root)
            or "No project instructions found."
        )

    return server
