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
    """Repository MCP 对外结构化的路径或内容命中。"""

    type: Literal["content", "path"]
    path: str
    line: int | None = None
    snippet: str | None = None


class RepositorySearchResult(BaseModel):
    """Repository 搜索命中、路由事实和扫描成本的稳定输出 schema。"""

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
    """创建只读 Repository Context MCP Server。

    Args:
        workspace: 唯一允许搜索和读取项目指令的仓库根。
        codecraft_home: RepositoryIndex 存储根的可选覆盖。

    Returns:
        暴露 ``search_repository`` tool、workspace metadata 和 project
        instructions 两个 resource 的 FastMCP 实例。

    Raises:
        ValueError: workspace 不是已存在目录。

    Server 默认同时配置 scan/lexical/symbol；索引不存在或陈旧时 Engine 会
    降级 scan，因此 MCP consumer 无需先知道本机索引状态。
    """
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
        """在 workspace 子目录内执行有界策略检索并返回完整结构化诊断。

        Example:
            MCP 调用参数 ``{"query": "Session", "path": "src", "strategy":
            "auto"}`` 会按查询形态选择 symbol/lexical/scan，并在响应记录实际
            retriever 与 attempted_retrievers。
        """
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
        """返回 workspace、索引是否存在及可用 Retriever 的稳定 JSON。"""
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
        """返回 workspace 根作用域的项目指令，缺失时返回明确文本。"""
        return (
            InstructionLoader().load_project_instructions(cwd=root)
            or "No project instructions found."
        )

    return server
