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
    def __init__(self, config: SessionConfig, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.config = config

    def set_config(self, config: SessionConfig) -> None:
        self.config = config
        self.refresh()

    def render(self) -> Text:
        return session_header(
            self.config,
            max(self.content_size.width, 1),
            palette_for(self.app.current_theme.dark),
        )


class RuntimeStatusLine(Static):
    def __init__(self, config: SessionConfig, *, id: str | None = None) -> None:
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
        self.config = config
        self.status = status
        self.token_usage = token_usage
        self.refresh()

    def render(self) -> Text:
        return runtime_status(
            self.config,
            self.status,
            self.token_usage,
            max(self.content_size.width, 1),
            palette_for(self.app.current_theme.dark),
        )


class MessageBlock(Static):
    def __init__(self, role: str, text: str = "") -> None:
        super().__init__(classes=role.casefold())
        self.role = role
        self.text = text

    def set_text(self, text: str) -> None:
        self.text = text
        self.refresh(layout=True)

    def render(self) -> RenderableType:
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
        super().__init__()
        self.call_id = call_id
        self.tool_name = name
        self.arguments = arguments or {}
        self.status = status
        self.duration_ms = duration_ms
        self.detail = detail

    @classmethod
    def notice(cls, text: str, *, failed: bool = False) -> ActivityBlock:
        return cls(text, status="failed" if failed else "notice")

    def mark_waiting(self) -> None:
        self.status = "waiting"
        self.refresh(layout=True)

    def mark_running(self) -> None:
        self.status = "running"
        self.refresh(layout=True)

    def mark_stopped(self) -> None:
        if self.status in {"running", "waiting"}:
            self.status = "stopped"
            self.refresh(layout=True)

    def finish(self, payload: EventPayload) -> None:
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
