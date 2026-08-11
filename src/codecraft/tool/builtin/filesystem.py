from __future__ import annotations

import asyncio
import difflib
from collections.abc import Generator
from itertools import islice
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from codecraft.core.errors import WorkspaceAccessError
from codecraft.retrieval.engine import ContextEngine
from codecraft.retrieval.models import RetrievalRequest
from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.tool.atomic import atomic_write_text
from codecraft.tool.base import BaseTool, ToolArguments, ToolContext
from codecraft.tool.workspace import WorkspaceGuard

_SKIPPED_ENTRY_NAMES = frozenset({".git", "__pycache__", ".venv", "node_modules"})


class ReadFileArgs(ToolArguments):
    """读取路径、字符编码和返回字符硬上限。"""

    path: str
    encoding: str = "utf-8"
    max_chars: int = Field(default=80_000, ge=1, le=1_000_000)


class ReadFileTool(BaseTool):
    """在 WorkspaceGuard 边界内有界读取文本文件的只读工具。"""

    name = "read_file"
    description = "Read a text file inside the workspace."
    args_schema = ReadFileArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """严格校验参数后在线程池执行阻塞文件读取。"""
        read_args = ReadFileArgs.model_validate(args)
        return await asyncio.to_thread(self._read_sync, read_args, context)

    @staticmethod
    def _read_sync(read_args: ReadFileArgs, context: ToolContext) -> ToolResult:
        """读取至 max_chars+1 以判断截断，并返回完整性和实际字节统计。

        成功 data 中的 line_count 只描述可见前缀；line_count_complete 明确告诉
        调用方它是否代表完整文件。解码、缺失、目录和 I/O 使用稳定错误码。
        """
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
            with path.open("r", encoding=read_args.encoding) as stream:
                file_bytes = os.fstat(stream.fileno()).st_size
                buffered = stream.read(read_args.max_chars + 1)
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
        except OSError as exc:
            return ToolResult(
                success=False,
                content="File could not be read.",
                error="file_read_error",
                metadata={"path": str(path), "reason": str(exc)},
            )

        truncated = len(buffered) > read_args.max_chars
        visible = buffered[: read_args.max_chars]
        metadata: dict[str, object] = {
            "path": str(path),
            "bytes": file_bytes,
            "returned_chars": len(visible),
            "truncated": truncated,
        }
        if not truncated:
            metadata["chars"] = len(visible)
        return ToolResult(
            success=True,
            content=visible,
            data={
                "path": str(path),
                "line_count": len(visible.splitlines()),
                "line_count_complete": not truncated,
                "truncated": truncated,
            },
            metadata=metadata,
        )


class WriteFileArgs(ToolArguments):
    """写入路径、完整替换文本、编码和是否创建父目录。"""

    path: str
    content: str
    encoding: str = "utf-8"
    create_parent_dirs: bool = False


class WriteFileTool(BaseTool):
    """审批后在 workspace 内原子创建或完整替换文本文件。"""

    name = "write_file"
    description = "Write text content to a file inside the workspace."
    args_schema = WriteFileArgs
    effects = {ToolEffect.WORKSPACE_WRITE}
    requires_approval = True

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """编码预检、读取旧值、原子写入并返回 changed/status/unified diff。

        内容相同不触碰文件，status=unchanged；真正变更使用同目录临时文件和
        os.replace，避免异常留下部分写入。编码失败发生在创建父目录之前。
        """
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

        try:
            encoded_size = len(write_args.content.encode(write_args.encoding))
        except (LookupError, UnicodeEncodeError) as exc:
            return ToolResult(
                success=False,
                content="File content could not be encoded.",
                error="file_encode_error",
                suggestion="Try a different encoding.",
                metadata={
                    "path": str(path),
                    "encoding": write_args.encoding,
                    "reason": str(exc),
                },
            )

        parent_error = self._ensure_parent(path, create=write_args.create_parent_dirs)
        if parent_error is not None:
            return parent_error

        try:
            previous = (
                path.read_text(encoding=write_args.encoding) if path.exists() else None
            )
        except (OSError, UnicodeError, LookupError) as exc:
            return ToolResult(
                success=False,
                content="Existing file could not be read before writing.",
                error="file_read_error",
                metadata={"path": str(path), "reason": str(exc)},
            )

        changed = previous != write_args.content
        status = "created" if previous is None else "modified"
        if not changed:
            status = "unchanged"
        else:
            try:
                atomic_write_text(
                    path,
                    write_args.content,
                    encoding=write_args.encoding,
                )
            except (OSError, UnicodeError, LookupError) as exc:
                return ToolResult(
                    success=False,
                    content="File could not be written.",
                    error="file_write_error",
                    metadata={"path": str(path), "reason": str(exc)},
                )

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
                "bytes": encoded_size,
            },
        )

    @staticmethod
    def _ensure_parent(path: Path, *, create: bool) -> ToolResult | None:
        """确认父目录存在，或按显式参数递归创建并归一化失败。"""
        if path.parent.exists():
            return None
        if not create:
            return ToolResult(
                success=False,
                content="Parent directory does not exist.",
                error="parent_directory_missing",
                suggestion="Set create_parent_dirs=true or create the parent directory first.",
                metadata={"path": str(path), "parent": str(path.parent)},
            )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                success=False,
                content="Parent directory could not be created.",
                error="parent_directory_create_failed",
                metadata={
                    "path": str(path),
                    "parent": str(path.parent),
                    "reason": str(exc),
                },
            )
        return None

    @staticmethod
    def _diff(*, before: str, after: str, path: Path) -> str:
        """生成以目标 basename 标记的 unified diff 审计事实。"""
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
        )


class ListFilesArgs(ToolArguments):
    """列表根、是否递归和最大可见条目数。"""

    path: str = "."
    recursive: bool = False
    max_entries: int = Field(default=500, ge=1, le=10_000)


class ListFilesTool(BaseTool):
    """确定性列出 workspace 内文件/目录且不递归 symlink 目录。"""

    name = "list_files"
    description = "List files and directories inside the workspace."
    args_schema = ListFilesArgs
    effects = {ToolEffect.READ_ONLY}

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """严格校验参数后在线程池执行阻塞目录遍历。"""
        list_args = ListFilesArgs.model_validate(args)
        return await asyncio.to_thread(self._list_sync, list_args, context)

    @classmethod
    def _list_sync(
        cls,
        list_args: ListFilesArgs,
        context: ToolContext,
    ) -> ToolResult:
        """多取一项判断截断，再排序、格式化成相对路径结果。"""
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
            visible_entries = [path]
            truncated = False
        else:
            entries = cls._iter_entries(path, recursive=list_args.recursive)
            try:
                visible_entries = list(islice(entries, list_args.max_entries + 1))
            except OSError as exc:
                return ToolResult(
                    success=False,
                    content="Directory could not be listed.",
                    error="directory_read_error",
                    metadata={"path": str(path), "reason": str(exc)},
                )
            finally:
                entries.close()
            truncated = len(visible_entries) > list_args.max_entries
            if truncated:
                visible_entries.pop()
            visible_entries.sort()

        lines = [cls._format_entry(entry, path) for entry in visible_entries]
        return ToolResult(
            success=True,
            content="\n".join(lines),
            data={
                "path": str(path),
                "entries": lines,
                "truncated": truncated,
            },
            metadata={
                "path": str(path),
                "count": len(lines),
                "recursive": list_args.recursive,
            },
        )

    @staticmethod
    def _iter_entries(path: Path, *, recursive: bool) -> Generator[Path, None, None]:
        """按名称深度优先遍历，跳过重目录且不跟随目录 symlink。"""
        with os.scandir(path) as directory:
            entries = sorted(directory, key=lambda item: item.name)
        for item in entries:
            if item.name in _SKIPPED_ENTRY_NAMES:
                continue
            entry = Path(item.path)
            yield entry
            if not recursive:
                continue
            try:
                is_directory = item.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_directory:
                yield from ListFilesTool._iter_entries(entry, recursive=True)

    @staticmethod
    def _format_entry(entry: Path, root: Path) -> str:
        """转换成相对 root 的展示路径，目录附加斜杠，逃逸则拒绝。"""
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
    """检索查询、作用域、匹配模式、策略及结果/文件预算。"""

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
    max_file_bytes: int = Field(default=1_000_000, ge=1, le=10_000_000)


class WorkspaceSearchTool(BaseTool):
    """把 ContextEngine 的扫描/索引/路由能力暴露为模型只读工具。"""

    name = "workspace_search"
    description = (
        "Search workspace paths, text, or indexed symbols with scan, lexical, or "
        "symbol retrieval, returning paths, line numbers, and snippets."
    )
    args_schema = WorkspaceSearchArgs
    effects = {ToolEffect.READ_ONLY}

    def __init__(self, context_engine: ContextEngine | None = None) -> None:
        """注入检索引擎；默认创建可直接扫描的 ContextEngine。"""
        self.context_engine = context_engine or ContextEngine()

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """约束 scope 后执行策略检索，并同时返回人类文本与结构化诊断。

        data/metadata 都保留 retriever、fallback_from、route_reason、attempted
        链和扫描成本，便于 ToolRunner、评测和 UI 在不同消费边界使用。
        """
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
        """把路径命中或内容命中格式化为紧凑可读单行。"""
        if match["type"] == "path":
            return f"{match['path']} [path]"
        return f"{match['path']}:{match['line']}: {match['snippet']}"
