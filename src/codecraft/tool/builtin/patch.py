from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from codecraft.core.errors import WorkspaceAccessError
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.tool import ToolEffect, ToolResult, ToolRuntimeEvent
from codecraft.tool.atomic import atomic_write_text
from codecraft.tool.base import BaseTool, ToolArguments, ToolContext
from codecraft.tool.workspace import WorkspaceGuard


class ApplyPatchArgs(ToolArguments):
    """通过工具协议传输的完整 unified diff 文本。"""

    patch: str


@dataclass(frozen=True)
class PatchFile:
    """解析后的目标相对路径及按出现顺序排列的 hunks。"""

    path: str
    hunks: list[list[str]]


@dataclass(frozen=True)
class PreparedPatch:
    """写入前冻结的目标绝对路径、旧内容与计算后内容。"""

    path: Path
    before: str
    after: str


class PatchApplicationError(Exception):
    """可直接转换为稳定 ToolResult 的 patch 业务失败。"""

    def __init__(
        self,
        *,
        content: str,
        error: str,
        metadata: dict[str, object] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """保存模型可见内容、错误码、建议和安全 metadata。"""
        super().__init__(content)
        self.content = content
        self.error = error
        self.metadata = metadata or {}
        self.suggestion = suggestion

    def as_result(self) -> ToolResult:
        """把异常字段无损转换为失败 ToolResult。"""
        return ToolResult(
            success=False,
            content=self.content,
            error=self.error,
            suggestion=self.suggestion,
            metadata=self.metadata,
        )


class ApplyPatchTool(BaseTool):
    """应用已有文件的 unified diff。

    当前实现只支持修改已存在文件，不支持新增/删除文件。patch 的路径仍会
    经过 WorkspaceGuard，避免 diff header 把写入目标带出 workspace。
    """

    name = "apply_patch"
    description = "Apply a unified diff patch to existing files inside the workspace."
    args_schema = ApplyPatchArgs
    effects = {ToolEffect.WORKSPACE_WRITE}
    requires_approval = True

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """解析 patch、逐文件应用 hunk，并返回变更文件列表。"""
        patch_args = ApplyPatchArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)

        try:
            files = self._parse_patch(patch_args.patch)
        except ValueError as exc:
            return ToolResult(
                success=False,
                content="Patch could not be parsed.",
                error="invalid_patch",
                suggestion="Provide a standard unified diff with ---/+++/@@ headers.",
                metadata={"reason": str(exc)},
            )

        try:
            prepared = self._prepare_changes(files, guard)
            self._verify_targets_unchanged(prepared)
            self._commit_changes(prepared)
        except PatchApplicationError as exc:
            return exc.as_result()

        changed_files = [str(change.path) for change in prepared]

        return ToolResult(
            success=True,
            content=f"applied patch to {len(changed_files)} file(s)",
            data={
                "changed_files": changed_files,
                "modified": len(changed_files),
                "added": 0,
                "deleted": 0,
                "diff": patch_args.patch,
            },
            metadata={
                "changed_files": changed_files,
                "modified": len(changed_files),
            },
            runtime_events=[
                ToolRuntimeEvent(
                    type=RuntimeEventType.PATCH_APPLIED,
                    payload={
                        "changed_files": changed_files,
                        "modified": len(changed_files),
                        "added": 0,
                        "deleted": 0,
                    },
                )
            ],
        )

    @classmethod
    def _prepare_changes(
        cls,
        files: list[PatchFile],
        guard: WorkspaceGuard,
    ) -> list[PreparedPatch]:
        """在不写磁盘的 prepare 阶段解析路径、读原文并应用全部 hunks。

        同一目标重复出现会拒绝；无实际变化的文件不会进入 commit 集合。
        WorkspaceAccessError 被翻译成 PatchApplicationError 统一返回。
        """
        prepared: list[PreparedPatch] = []
        seen: set[Path] = set()

        for patch_file in files:
            try:
                path = guard.resolve_write_path(patch_file.path)
            except WorkspaceAccessError as exc:
                raise PatchApplicationError(
                    content=exc.message,
                    error=exc.code,
                    suggestion=exc.suggestion,
                    metadata=exc.metadata,
                ) from exc

            if path in seen:
                raise PatchApplicationError(
                    content="Patch contains the same target more than once.",
                    error="duplicate_patch_target",
                    metadata={"path": str(path)},
                )
            seen.add(path)
            cls._validate_target(path)

            try:
                before = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PatchApplicationError(
                    content="Patch target could not be read.",
                    error="patch_read_failed",
                    metadata={"path": str(path), "reason": str(exc)},
                ) from exc

            try:
                after = cls._apply_hunks(before, patch_file.hunks)
            except ValueError as exc:
                raise PatchApplicationError(
                    content="Patch could not be applied.",
                    error="patch_conflict",
                    metadata={"path": str(path), "reason": str(exc)},
                ) from exc

            if before != after:
                prepared.append(PreparedPatch(path=path, before=before, after=after))

        return prepared

    @staticmethod
    def _validate_target(path: Path) -> None:
        """当前版本只接受已存在的普通文件，不支持新增、删除或目录。"""
        if not path.exists():
            raise PatchApplicationError(
                content="Patch target does not exist.",
                error="patch_target_missing",
                metadata={"path": str(path)},
            )
        if path.is_dir():
            raise PatchApplicationError(
                content="Patch target is a directory.",
                error="path_is_directory",
                metadata={"path": str(path)},
            )

    @staticmethod
    def _verify_targets_unchanged(prepared: list[PreparedPatch]) -> None:
        """commit 前重读全部目标，发现 prepare 后并发变化则整体拒绝。"""
        for change in prepared:
            try:
                current = change.path.read_text(encoding="utf-8")
            except OSError as exc:
                raise PatchApplicationError(
                    content="Patch target changed before it could be written.",
                    error="patch_target_changed",
                    metadata={"path": str(change.path), "reason": str(exc)},
                ) from exc
            if current != change.before:
                raise PatchApplicationError(
                    content="Patch target changed before it could be written.",
                    error="patch_target_changed",
                    metadata={"path": str(change.path)},
                )

    @staticmethod
    def _commit_changes(prepared: list[PreparedPatch]) -> None:
        """依次原子替换；中途失败时逆序尽力回滚已提交文件。

        若回滚也失败，metadata 明确列出 rollback_errors 和
        possibly_changed_files，提醒调用方先检查真实状态再重试。
        """
        committed: list[PreparedPatch] = []
        try:
            for change in prepared:
                atomic_write_text(change.path, change.after)
                committed.append(change)
        except Exception as exc:
            rollback_errors: list[dict[str, str]] = []
            for change in reversed(committed):
                try:
                    atomic_write_text(change.path, change.before)
                except Exception as rollback_exc:
                    rollback_errors.append(
                        {"path": str(change.path), "reason": str(rollback_exc)}
                    )
            metadata: dict[str, object] = {"reason": str(exc)}
            if rollback_errors:
                metadata["rollback_errors"] = rollback_errors
                metadata["possibly_changed_files"] = [
                    str(change.path) for change in committed
                ]
            raise PatchApplicationError(
                content="Patch changes could not be committed.",
                error="patch_write_failed",
                metadata=metadata,
                suggestion=(
                    "Inspect possibly_changed_files before retrying."
                    if rollback_errors
                    else "Fix the filesystem error and retry the patch."
                ),
            ) from exc

    @staticmethod
    def _parse_patch(patch: str) -> list[PatchFile]:
        """从 unified diff 中提取文件路径和 hunk。"""
        # The tool argument is a transport string, so its final record may omit the
        # transport newline. Unified diff uses an explicit marker when file content
        # itself has no trailing newline.
        normalized_patch = patch if patch.endswith("\n") else f"{patch}\n"
        lines = normalized_patch.splitlines(keepends=True)
        files: list[PatchFile] = []
        index = 0

        while index < len(lines):
            if not lines[index].startswith("--- "):
                index += 1
                continue
            patch_file, index = ApplyPatchTool._parse_patch_file(lines, index)
            files.append(patch_file)

        if not files:
            raise ValueError("patch contains no file changes")
        return files

    @staticmethod
    def _parse_patch_file(lines: list[str], index: int) -> tuple[PatchFile, int]:
        """从一个 ---/+++ header 读取目标路径及至少一个 hunk。"""
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise ValueError("missing +++ file header")

        path = ApplyPatchTool._normalize_patch_path(lines[index + 1][4:].strip())
        index += 2
        hunks: list[list[str]] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@"):
                index += 1
                continue
            hunk, index = ApplyPatchTool._parse_hunk(lines, index)
            hunks.append(hunk)

        if not hunks:
            raise ValueError(f"patch for {path} has no hunks")
        return PatchFile(path=path, hunks=hunks), index

    @staticmethod
    def _parse_hunk(lines: list[str], index: int) -> tuple[list[str], int]:
        """读取 @@ header 后受支持的 context/add/remove/newline records。"""
        hunk = [lines[index]]
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.startswith("@@") or line.startswith("--- "):
                break
            if not line.startswith(("+", "-", " ", "\\")):
                raise ValueError(f"unsupported patch line: {line.rstrip()}")
            hunk.append(line)
            index += 1
        return hunk, index

    @staticmethod
    def _normalize_patch_path(raw_path: str) -> str:
        """移除时间戳与 a/b 前缀，并拒绝 /dev/null 新增删除语义。"""
        if raw_path == "/dev/null":
            raise ValueError("creating or deleting files is not supported yet")

        path = raw_path.split("\t", 1)[0].split(" ", 1)[0]
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if not path:
            raise ValueError("empty patch path")
        return path

    @staticmethod
    def _apply_hunks(content: str, hunks: list[list[str]]) -> str:
        """按 hunk header 的旧文件位置应用增删行。"""
        source = content.splitlines(keepends=True)
        result: list[str] = []
        cursor = 0

        for hunk in hunks:
            old_start = ApplyPatchTool._old_start_from_header(hunk[0])
            target_index = old_start - 1
            if target_index < cursor:
                raise ValueError("overlapping hunks")

            # 先复制 hunk 之前未触碰的原文，再根据前缀处理上下文/删除/新增行。
            result.extend(source[cursor:target_index])
            cursor = target_index

            for prefix, body in ApplyPatchTool._hunk_records(hunk[1:]):
                if prefix == " ":
                    if cursor >= len(source) or source[cursor] != body:
                        raise ValueError("patch context does not match")
                    result.append(source[cursor])
                    cursor += 1
                elif prefix == "-":
                    if cursor >= len(source) or source[cursor] != body:
                        raise ValueError("patch removal does not match")
                    cursor += 1
                elif prefix == "+":
                    result.append(body)

        result.extend(source[cursor:])
        return "".join(result)

    @staticmethod
    def _hunk_records(lines: list[str]) -> Iterator[tuple[str, str]]:
        """解释 hunk 行前缀及标准 no-newline marker，产生 prefix/body。"""
        index = 0
        while index < len(lines):
            line = lines[index]
            prefix = line[:1]
            if prefix == "\\":
                raise ValueError("newline marker has no preceding patch line")

            body = line[1:]
            if index + 1 < len(lines) and lines[index + 1].startswith("\\"):
                marker = lines[index + 1].rstrip("\r\n")
                if marker != "\\ No newline at end of file":
                    raise ValueError(f"unsupported patch marker: {marker}")
                body = body.removesuffix("\n").removesuffix("\r")
                index += 1

            yield prefix, body
            index += 1

    @staticmethod
    def _old_start_from_header(header: str) -> int:
        """从 ``@@ -old_start,count +... @@`` 提取旧文件 1-based 起始行。"""
        try:
            old_range = header.split(" ", 2)[1]
            start = old_range.removeprefix("-").split(",", 1)[0]
            return int(start)
        except Exception as exc:
            raise ValueError(f"invalid hunk header: {header.rstrip()}") from exc
