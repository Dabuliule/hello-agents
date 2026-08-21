from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from codecraft.sandbox import (
    ProcessSandboxBackend,
    SandboxBackend,
    SandboxBackendError,
    SandboxExecutionRequest,
    SandboxExecutionResult,
)
from codecraft.sandbox.command_policy import CommandRisk
from codecraft.schema.tool import ToolEffect, ToolResult
from codecraft.tool.base import BaseTool, ToolArguments, ToolContext
from codecraft.tool.workspace import WorkspaceGuard


class BashArgs(ToolArguments):
    """Shell 文本、可选 workspace 子目录 cwd 和进程超时。"""

    command: str
    cwd: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class BashTool(BaseTool):
    """在 workspace 内执行 shell command 的内置工具。

    命令先由 ApprovalManager 调用 CommandPolicy 分类，再随 ToolContext 到达这里。
    BashTool 在真正启动进程前再次拒绝 DENY，以及“需要审批但没有 approved”的
    PROMPT 命令；因此 approval_policy=never 不会把高风险命令变成自动允许。通过
    检查后才交给 SandboxBackend，并把输出、退出码和后端错误归一化为 ToolResult。
    """

    name = "bash"
    description = "Run a shell command from inside the workspace."
    args_schema = BashArgs
    effects = {ToolEffect.PROCESS_EXEC}
    requires_approval = True

    def __init__(
        self,
        sandbox_backend: SandboxBackend | None = None,
    ) -> None:
        """注入隔离后端；直接构造时默认使用明确标注无隔离的 Process 后端。"""
        self.sandbox_backend = sandbox_backend or ProcessSandboxBackend()

    async def arun(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """执行命令并返回 stdout/stderr、exit code 和截断信息。"""
        bash_args = BashArgs.model_validate(args)
        guard = WorkspaceGuard(context.context.cwd)
        cwd = self._resolve_cwd(bash_args.cwd, context, guard)
        decision = context.command_decision
        if decision is None:
            return ToolResult(
                success=False,
                content="Command policy was not evaluated.",
                error="command_policy_missing",
            )

        if decision.risk == CommandRisk.DENY:
            return ToolResult(
                success=False,
                content="Command denied by policy.",
                error="command_denied",
                suggestion=decision.reason,
                metadata={"command": bash_args.command, "risk": decision.risk},
            )

        if decision.requires_approval and not context.approved:
            return ToolResult(
                success=False,
                content="Command requires approval.",
                error="command_requires_approval",
                suggestion=decision.reason,
                metadata={"command": bash_args.command, "risk": decision.risk},
            )

        try:
            execution = await self.sandbox_backend.execute(
                SandboxExecutionRequest(
                    command=bash_args.command,
                    cwd=cwd,
                    workspace_root=context.context.cwd,
                    sandbox_mode=context.context.sandbox_mode,
                    network_access=context.context.network_access,
                    timeout_seconds=bash_args.timeout_seconds,
                    max_output_bytes=min(
                        context.context.max_tool_output_chars * 4,
                        40_000_000,
                    ),
                    env_allowlist=tuple(context.context.sandbox_env_allowlist),
                    allow_workspace_path_entries=context.approved,
                )
            )
        except SandboxBackendError as exc:
            return ToolResult(
                success=False,
                content="Sandbox backend could not execute the command.",
                error="sandbox_backend_error",
                suggestion=str(exc),
                metadata={
                    "command": bash_args.command,
                    "cwd": str(cwd),
                    "risk": decision.risk,
                    "backend": self.sandbox_backend.name,
                },
            )

        stdout = execution.stdout.decode("utf-8", errors="replace")
        stderr = execution.stderr.decode("utf-8", errors="replace")
        stdout, stdout_truncated = self._truncate(
            stdout, context.context.max_tool_output_chars
        )
        stderr, stderr_truncated = self._truncate(
            stderr, context.context.max_tool_output_chars
        )
        stdout_truncated = execution.stdout_truncated or stdout_truncated
        stderr_truncated = execution.stderr_truncated or stderr_truncated
        exit_code = execution.exit_code
        success = (
            exit_code == 0
            and not execution.timed_out
            and execution.backend_error is None
        )

        return ToolResult(
            success=success,
            content=stdout if success else stderr or stdout or "Command failed.",
            data={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": execution.timed_out,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            },
            error=None if success else _execution_error(execution),
            metadata={
                "command": bash_args.command,
                "cwd": str(cwd),
                "risk": decision.risk,
                **execution.metadata,
            },
        )

    @staticmethod
    def _resolve_cwd(
        cwd: str | None,
        context: ToolContext,
        guard: WorkspaceGuard,
    ) -> Path:
        """解析命令工作目录，确保 cwd 是 workspace 内的目录。"""
        if cwd is None:
            return context.context.cwd
        resolved = guard.resolve_read_path(cwd)
        if not resolved.is_dir():
            raise NotADirectoryError(str(resolved))
        return resolved

    @staticmethod
    def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
        """保留字符串前 max_chars 字符并返回是否截断。"""
        if len(value) <= max_chars:
            return value, False
        return value[:max_chars], True


def _execution_error(execution: SandboxExecutionResult) -> str:
    """按超时、后端启动错误、命令非零的优先级生成稳定错误码。"""
    if execution.timed_out:
        return "command_timed_out"
    if execution.backend_error is not None:
        return "sandbox_backend_error"
    return "command_failed"
