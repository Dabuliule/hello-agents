from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionSummary


class SessionRenderer:
    """渲染 Session 列表、Inspect 事件及错误/工具筛选视图。"""

    def __init__(self, console: Console) -> None:
        """绑定输出 Console。"""
        self.console = console

    def render_sessions(self, summaries: Iterable[SessionSummary]) -> None:
        """表格展示 Session ID、来源、事件数、有效性与紧凑 cwd。"""
        table = Table(title="Recent Sessions")
        table.add_column("Session")
        table.add_column("Source")
        table.add_column("Events", justify="right")
        table.add_column("Status")
        table.add_column("CWD")
        for summary in summaries:
            status = self.status_text(summary)
            table.add_row(
                summary.session_id,
                str(summary.source or "-"),
                str(summary.event_count),
                status,
                shorten_path(summary.cwd),
            )
        self.console.print(table)

    def render_inspect_summary(
        self, session_id: str, events: list[RuntimeEvent]
    ) -> None:
        """展示指定日志的事件数、末事件和是否存在最终 answer。"""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        table.add_row("session", session_id)
        table.add_row("events", str(len(events)))
        table.add_row("last event", str(events[-1].type) if events else "-")
        table.add_row("final answer", "available" if last_answer(events) else "-")
        self.console.print(Panel(table, title="session inspect", border_style="cyan"))

    def render_events(self, events: list[RuntimeEvent]) -> None:
        """按 seq 表格展示全部事件的类型、Turn 和短摘要。"""
        table = Table(title="Events")
        table.add_column("Seq", justify="right")
        table.add_column("Type")
        table.add_column("Turn")
        table.add_column("Summary")
        for event in events:
            table.add_row(
                str(event.seq),
                str(event.type),
                event.turn_id or "-",
                event_summary(event),
            )
        self.console.print(table)

    def render_tool_events(self, events: list[RuntimeEvent]) -> None:
        """筛选模型请求/开始/完成 Tool 事件并显示状态与预览。"""
        table = Table(title="Tool Events")
        table.add_column("Seq", justify="right")
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Duration")
        table.add_column("Preview")
        for event in events:
            if event.type == RuntimeEventType.MODEL_TOOL_CALL:
                table.add_row(
                    str(event.seq),
                    str(event.payload.get("name") or "-"),
                    "requested",
                    "-",
                    f"args={event.payload.get('arguments')}",
                )
            elif event.type == RuntimeEventType.TOOL_CALL_STARTED:
                table.add_row(
                    str(event.seq),
                    str(event.payload.get("name") or "-"),
                    "started",
                    "-",
                    "",
                )
            elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
                result = event.payload.get("result")
                success = isinstance(result, dict) and result.get("success") is True
                preview = ""
                if isinstance(result, dict):
                    preview = str(result.get("content") or result.get("error") or "")
                table.add_row(
                    str(event.seq),
                    str(event.payload.get("name") or "-"),
                    "ok" if success else "failed",
                    f"{event.payload.get('duration_ms')}ms"
                    if isinstance(event.payload.get("duration_ms"), int)
                    else "-",
                    " ".join(preview.split())[:160],
                )
        self.console.print(table)

    def render_error_events(self, events: list[RuntimeEvent]) -> None:
        """筛选 ERROR、TURN_ABORTED 和失败 ToolResult。"""
        table = Table(title="Errors")
        table.add_column("Seq", justify="right")
        table.add_column("Type")
        table.add_column("Turn")
        table.add_column("Payload")
        for event in events:
            if event.type in {RuntimeEventType.ERROR, RuntimeEventType.TURN_ABORTED}:
                table.add_row(
                    str(event.seq),
                    str(event.type),
                    event.turn_id or "-",
                    str(event.payload.get("message") or event.payload),
                )
            elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
                result = event.payload.get("result")
                if isinstance(result, dict) and result.get("success") is False:
                    table.add_row(
                        str(event.seq), "tool_error", event.turn_id or "-", str(result)
                    )
        self.console.print(table)

    @staticmethod
    def status_text(summary: SessionSummary) -> str:
        """返回 valid 或带稳定 error code 的 invalid 状态。"""
        return "valid" if summary.valid else f"invalid:{summary.error_code or '-'}"


def shorten_path(path: Path | None, *, max_chars: int = 48) -> str:
    """保留路径末尾的最长可见部分，缺失路径显示短横线。"""
    if path is None:
        return "-"
    text = str(path)
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars + 3 :]


def last_answer(events: list[RuntimeEvent]) -> str | None:
    """反向读取最近 TURN_FINISHED 的字符串 answer。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_FINISHED:
            answer = event.payload.get("answer")
            if isinstance(answer, str):
                return answer
    return None


def event_summary(event: RuntimeEvent) -> str:
    """按类型从 payload 选取最多 120 字符的检查视图摘要。"""
    payload = event.payload
    if event.type in {
        RuntimeEventType.ASSISTANT_MESSAGE,
        RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
    }:
        return str(payload.get("text") or "")[:120]
    if event.type == RuntimeEventType.MODEL_TOOL_CALL:
        return f"{payload.get('name')} args={payload.get('arguments')}"
    if event.type == RuntimeEventType.TOOL_CALL_FINISHED:
        result = payload.get("result")
        if isinstance(result, dict):
            return str(result.get("content") or result.get("error") or "")[:120]
    if event.type in {RuntimeEventType.ERROR, RuntimeEventType.TURN_ABORTED}:
        return str(payload.get("message") or payload)
    return str(payload)[:120]
