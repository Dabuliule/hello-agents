from __future__ import annotations

import difflib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from codecraft.core.errors import WorkspaceAccessError
from codecraft.retrieval.engine import ContextEngine
from codecraft.retrieval.models import RetrievalRequest
from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.tool.base import BaseTool, ToolArguments, ToolContext
from codecraft.tool.workspace import WorkspaceGuard


class ReadFileArgs(ToolArguments):
    path: str
    encoding: str = "utf-8"
    max_chars: int = Field(default=80_000, ge=1)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a text file inside the workspace."
    args_schema = ReadFileArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        read_args = ReadFileArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)
        path = guard.resolve_read_path(read_args.path)

        if path.is_dir():
            return ToolResult(
                success=False,
                content="Cannot read a directory.",
                error="path_is_directory",
                metadata={"path": str(path)},
            )

        try:
            content = path.read_text(encoding=read_args.encoding)
        except FileNotFoundError:
            return ToolResult(
                success=False,
                content="File does not exist.",
                error="file_not_found",
                metadata={"path": str(path)},
            )
        except UnicodeDecodeError as exc:
            return ToolResult(
                success=False,
                content="File could not be decoded.",
                error="file_decode_error",
                suggestion="Try a different encoding.",
                metadata={
                    "path": str(path),
                    "encoding": read_args.encoding,
                    "reason": str(exc),
                },
            )

        truncated = len(content) > read_args.max_chars
        visible = content[: read_args.max_chars]
        return ToolResult(
            success=True,
            content=visible,
            data={
                "path": str(path),
                "line_count": len(content.splitlines()),
                "truncated": truncated,
            },
            metadata={
                "path": str(path),
                "chars": len(content),
                "returned_chars": len(visible),
                "truncated": truncated,
            },
        )


class WriteFileArgs(ToolArguments):
    path: str
    content: str
    encoding: str = "utf-8"
    create_parent_dirs: bool = False


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write text content to a file inside the workspace."
    args_schema = WriteFileArgs
    effects = {ToolEffect.WORKSPACE_WRITE}
    requires_approval = True

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        write_args = WriteFileArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)
        path = guard.resolve_write_path(write_args.path)

        if path.exists() and path.is_dir():
            return ToolResult(
                success=False,
                content="Cannot write file because path is a directory.",
                error="path_is_directory",
                metadata={"path": str(path)},
            )

        if not path.parent.exists():
            if not write_args.create_parent_dirs:
                return ToolResult(
                    success=False,
                    content="Parent directory does not exist.",
                    error="parent_directory_missing",
                    suggestion="Set create_parent_dirs=true or create the parent directory first.",
                    metadata={"path": str(path), "parent": str(path.parent)},
                )
            path.parent.mkdir(parents=True, exist_ok=True)

        previous = (
            path.read_text(encoding=write_args.encoding) if path.exists() else None
        )
        path.write_text(write_args.content, encoding=write_args.encoding)

        status = "created" if previous is None else "modified"
        changed = previous != write_args.content
        diff = self._diff(
            before=previous or "",
            after=write_args.content,
            path=path,
        )
        return ToolResult(
            success=True,
            content=f"{status} {path}",
            data={
                "path": str(path),
                "status": status,
                "changed": changed,
                "diff": diff,
            },
            metadata={
                "path": str(path),
                "status": status,
                "changed": changed,
                "bytes": len(write_args.content.encode(write_args.encoding)),
            },
        )

    @staticmethod
    def _diff(*, before: str, after: str, path: Path) -> str:
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
        )


class ListFilesArgs(ToolArguments):
    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=500, ge=1)


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List files and directories inside the workspace."
    args_schema = ListFilesArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        list_args = ListFilesArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)
        path = guard.resolve_read_path(list_args.path)

        if not path.exists():
            return ToolResult(
                success=False,
                content="Path does not exist.",
                error="path_not_found",
                metadata={"path": str(path)},
            )

        if path.is_file():
            entries = [path]
        else:
            entries = self._iter_entries(path, recursive=list_args.recursive)

        visible_entries = []
        skipped_names = {".git", "__pycache__", ".venv", "node_modules"}
        for entry in entries:
            if any(part in skipped_names for part in entry.relative_to(path).parts):
                continue
            visible_entries.append(entry)
            if len(visible_entries) >= list_args.max_entries:
                break

        lines = [self._format_entry(entry, path) for entry in visible_entries]
        return ToolResult(
            success=True,
            content="\n".join(lines),
            data={
                "path": str(path),
                "entries": lines,
                "truncated": len(visible_entries) >= list_args.max_entries,
            },
            metadata={
                "path": str(path),
                "count": len(lines),
                "recursive": list_args.recursive,
            },
        )

    @staticmethod
    def _iter_entries(path: Path, *, recursive: bool) -> list[Path]:
        if recursive:
            return sorted(path.rglob("*"))
        return sorted(path.iterdir())

    @staticmethod
    def _format_entry(entry: Path, root: Path) -> str:
        suffix = "/" if entry.is_dir() else ""
        try:
            relative = entry.relative_to(root)
        except ValueError as exc:
            raise WorkspaceAccessError(
                "listed path escaped root",
                code="workspace_access_denied",
            ) from exc
        return f"{relative}{suffix}"


class WorkspaceSearchArgs(ToolArguments):
    query: str = Field(min_length=1)
    path: str = "."
    mode: Literal["both", "content", "path"] = "both"
    case_sensitive: bool = False
    strategy: Literal["auto", "scan", "lexical", "symbol"] = Field(
        default="auto",
        description=(
            "auto to route by query shape, scan for exact path/text matching, "
            "lexical for ranked indexed search, or symbol for indexed definitions"
        ),
    )
    max_results: int = Field(default=100, ge=1, le=1000)
    max_file_bytes: int = Field(default=1_000_000, ge=1)


class WorkspaceSearchTool(BaseTool):
    name = "workspace_search"
    description = (
        "Search workspace paths, text, or indexed symbols with scan, lexical, or "
        "symbol retrieval, returning paths, line numbers, and snippets."
    )
    args_schema = WorkspaceSearchArgs
    effects = {ToolEffect.READ_ONLY}

    def __init__(self, context_engine: ContextEngine | None = None) -> None:
        self.context_engine = context_engine or ContextEngine()

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        search_args = WorkspaceSearchArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)
        root = guard.resolve_read_path(search_args.path)

        if not root.exists():
            return ToolResult(
                success=False,
                content="Search path does not exist.",
                error="path_not_found",
                metadata={"path": str(root)},
            )

        response = await self.context_engine.retrieve(
            RetrievalRequest(
                query=search_args.query,
                root=root,
                workspace_root=context.context.cwd,
                mode=search_args.mode,
                case_sensitive=search_args.case_sensitive,
                max_results=search_args.max_results,
                max_file_bytes=search_args.max_file_bytes,
            ),
            retriever_name=search_args.strategy,
            fallback_retriever="scan",
        )
        matches = [match.as_dict() for match in response.matches]
        stats = response.stats

        lines = [self._format_match(match) for match in matches]
        content = "\n".join(lines) if lines else "No matches found."

        return ToolResult(
            success=True,
            content=content,
            data={
                "query": search_args.query,
                "path": str(root),
                "matches": matches,
                "match_count": len(matches),
                "truncated": response.truncated,
                "skipped": stats.skipped,
                "candidate_file_count": stats.candidate_file_count,
                "scanned_file_count": stats.scanned_file_count,
                "read_file_count": stats.read_file_count,
                "scanned_bytes": stats.scanned_bytes,
                "returned_chars": len(content),
                "retriever": response.retriever,
                "fallback_from": response.fallback_from,
                "route_reason": response.route_reason,
                "attempted_retrievers": list(response.attempted_retrievers),
            },
            metadata={
                "query": search_args.query,
                "path": str(root),
                "match_count": len(matches),
                "truncated": response.truncated,
                "candidate_file_count": stats.candidate_file_count,
                "scanned_file_count": stats.scanned_file_count,
                "read_file_count": stats.read_file_count,
                "scanned_bytes": stats.scanned_bytes,
                "returned_chars": len(content),
                "retriever": response.retriever,
                "fallback_from": response.fallback_from,
                "route_reason": response.route_reason,
                "attempted_retrievers": list(response.attempted_retrievers),
            },
        )

    @staticmethod
    def _format_match(match: dict[str, object]) -> str:
        if match["type"] == "path":
            return f"{match['path']} [path]"
        return f"{match['path']}:{match['line']}: {match['snippet']}"
