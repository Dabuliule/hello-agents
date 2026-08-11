from __future__ import annotations

from typing import Any


class CodecraftError(Exception):
    """跨 Runtime 边界可安全展示的稳定错误码、建议和结构化上下文。"""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        suggestion: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """保存用户消息与机器字段，同时初始化标准 Exception 文本。"""
        self.message = message
        self.code = code
        self.suggestion = suggestion
        self.metadata = metadata or {}
        super().__init__(message)


class SessionError(CodecraftError):
    """Session 输入、状态或调度操作无效。"""


class SessionRestoreError(CodecraftError):
    """持久化日志无法安全恢复为 SessionSnapshot。"""


class ToolNotFoundError(CodecraftError):
    """ToolRegistry 中不存在模型请求的工具。"""


class ToolExecutionError(CodecraftError):
    """工具执行阶段的领域失败。"""


class WorkspaceAccessError(CodecraftError):
    """文件路径为空或解析后逃逸 workspace。"""


class ApprovalDeniedError(CodecraftError):
    """用户或 Reviewer 拒绝副作用授权。"""


class ModelProviderError(CodecraftError):
    """Provider 请求、协议或流式终态无效。"""


class CommandDeniedError(CodecraftError):
    """Shell 命令被静态安全策略硬拒绝。"""
