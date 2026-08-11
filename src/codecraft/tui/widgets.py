from __future__ import annotations

import json
from typing import Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static

from codecraft.schema.session import SessionConfig
from codecraft.tui.rendering import runtime_status, session_header
from codecraft.schema.event import EventPayload
from codecraft.tui.theme import palette_for


class SessionHeader(Static):
    """随宽度和 Theme 自适应显示 workspace/model 的顶部 Header。"""

    def __init__(self, config: SessionConfig, *, id: str | None = None) -> None:
        """保存当前 SessionConfig。"""
        super().__init__(id=id)
        self.config = config

    def set_config(self, config: SessionConfig) -> None:
        """替换恢复后的配置并请求重绘。"""
        self.config = config
        self.refresh()

    def render(self) -> Text:
        """用当前内容宽度和 Theme Palette 生成 Header Text。"""
        return session_header(
            self.config,
            max(self.content_size.width, 1),
            palette_for(self.app.current_theme.dark),
        )


class RuntimeStatusLine(Static):
    """显示 Turn 状态、Sandbox、Token 和 MCP 数量的底部状态行。"""

    def __init__(self, config: SessionConfig, *, id: str | None = None) -> None:
        """初始化 starting 状态与零 Token。"""
        super().__init__(id=id)
        self.config = config
        self.status = "starting"
        self.token_usage = {"total_tokens": 0}

    def set_state(
        self,
        config: SessionConfig,
        status: str,
        token_usage: dict[str, int],
    ) -> None:
        """原子替换状态栏输入并请求重绘。"""
        self.config = config
        self.status = status
        self.token_usage = token_usage
        self.refresh()

    def render(self) -> Text:
        """按当前宽度和 Theme 渲染自适应状态 Text。"""
        return runtime_status(
            self.config,
            self.status,
            self.token_usage,
            max(self.content_size.width, 1),
            palette_for(self.app.current_theme.dark),
        )


class MessageBlock(Static):
    """按 Assistant/User/Error/其他角色渲染对话文本的 Widget。"""

    def __init__(self, role: str, text: str = "") -> None:
        """保存角色/正文，并以角色小写设置 CSS class。"""
        super().__init__(classes=role.casefold())
        self.role = role
        self.text = text

    def set_text(self, text: str) -> None:
        """更新流式/完整正文并请求布局级刷新。"""
        self.text = text
        self.refresh(layout=True)

    def render(self) -> RenderableType:
        """Assistant 用 Markdown，User/Error 用图标语义色，其余用 Group。"""
        palette = palette_for(self.app.current_theme.dark)
        if self.role == "Assistant":
            return Markdown(self.text)

        if self.role == "User":
            body = Text()
            body.append("› ", style=f"bold {palette.accent}")
            body.append(self.text, style=palette.strong)
            return body

        if self.role == "Error":
            body = Text()
            body.append("! ", style=f"bold {palette.error}")
            body.append(self.text, style=palette.error_detail)
            return body

        return Group(
            Text(self.role.casefold(), style=f"bold {palette.muted}"),
            Text(self.text, style=palette.secondary),
        )


class ActivityBlock(Static):
    """对话流中的一条工具或运行状态记录。"""

    def __init__(
        self,
        name: str,
        *,
        call_id: str | None = None,
        arguments: dict[str, Any] | None = None,
        status: str = "running",
        duration_ms: int | None = None,
        detail: str = "",
    ) -> None:
        """初始化 Tool/notice 身份、参数、状态、耗时和详情。"""
        super().__init__()
        self.call_id = call_id
        self.tool_name = name
        self.arguments = arguments or {}
        self.status = status
        self.duration_ms = duration_ms
        self.detail = detail

    @classmethod
    def notice(cls, text: str, *, failed: bool = False) -> ActivityBlock:
        """创建普通或失败的非 Tool 活动提示。"""
        return cls(text, status="failed" if failed else "notice")

    def mark_waiting(self) -> None:
        """将 Tool 标为等待审批并刷新。"""
        self.status = "waiting"
        self.refresh(layout=True)

    def mark_running(self) -> None:
        """将审批后的 Tool 恢复为运行并刷新。"""
        self.status = "running"
        self.refresh(layout=True)

    def mark_stopped(self) -> None:
        """只把尚在 running/waiting 的残留活动标为 stopped。"""
        if self.status in {"running", "waiting"}:
            self.status = "stopped"
            self.refresh(layout=True)

    def finish(self, payload: EventPayload) -> None:
        """从 TOOL_CALL_FINISHED 设置成功/失败、合法耗时和最多 500 字错误详情。"""
        result = payload.get("result")
        success = isinstance(result, dict) and result.get("success") is True
        self.status = "completed" if success else "failed"
        duration = payload.get("duration_ms")
        self.duration_ms = (
            duration
            if isinstance(duration, int)
            and not isinstance(duration, bool)
            and duration >= 0
            else None
        )
        if not success and isinstance(result, dict):
            content = result.get("content") or result.get("error")
            if content:
                self.detail = str(content)[:500]
        self.refresh(layout=True)

    def render(self) -> Text:
        """按状态选择 symbol/colors，并按宽度显示参数、耗时与失败详情。"""
        palette = palette_for(self.app.current_theme.dark)
        symbol, symbol_style = {
            "running": ("·", f"bold {palette.accent}"),
            "waiting": ("?", f"bold {palette.warning}"),
            "completed": ("✓", f"bold {palette.success}"),
            "failed": ("×", f"bold {palette.error}"),
            "stopped": ("×", palette.muted),
            "notice": ("·", palette.muted),
        }.get(self.status, ("·", palette.muted))

        line = Text()
        line.append(f"{symbol} ", style=symbol_style)
        name_style = (
            palette.secondary if self.status == "notice" else palette.foreground
        )
        line.append(self.tool_name, style=name_style)

        arguments = _compact_arguments(self.arguments)
        if arguments:
            line.append("  ·  ", style=palette.separator)
            line.append(arguments, style=palette.muted)
        if self.duration_ms is not None and self.content_size.width >= 50:
            line.append("  ·  ", style=palette.separator)
            line.append(f"{self.duration_ms}ms", style=palette.faint)
        if self.detail:
            detail_style = (
                palette.error_detail if self.status == "failed" else palette.muted
            )
            line.append(f"\n  {self.detail}", style=detail_style)
        return line


def _compact_arguments(arguments: dict[str, Any]) -> str:
    """优先显示 path/command/query/pattern/url，否则稳定 JSON，最多 140 字符。"""
    if not arguments:
        return ""
    for key in ("path", "command", "query", "pattern", "url"):
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            compact_value = " ".join(str(item) for item in value)
        elif isinstance(value, dict):
            compact_value = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            compact_value = str(value)
        return (
            compact_value if len(compact_value) <= 140 else compact_value[:137] + "..."
        )
    compact = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return compact if len(compact) <= 140 else compact[:137] + "..."
