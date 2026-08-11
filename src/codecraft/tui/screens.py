from __future__ import annotations

import json
from typing import Any

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Label, Static

from codecraft.schema.session import SessionSummary
from codecraft.tui.theme import palette_for


class SessionBrowserScreen(ModalScreen[str | None]):
    """启动时选择最近 Session 或明确新建的 Modal Screen。"""

    BINDINGS = [Binding("escape", "new_session", show=False)]

    def __init__(self, summaries: list[SessionSummary]) -> None:
        """保存按最近时间排列的可恢复摘要。"""
        super().__init__()
        self.summaries = summaries

    def compose(self) -> ComposeResult:
        """声明 Session DataTable 和 New/Resume actions。"""
        with Vertical(id="session-dialog"):
            yield Label("Sessions", id="session-title")
            yield DataTable(
                id="session-table",
                cursor_type="row",
                zebra_stripes=True,
            )
            with Horizontal(id="session-actions"):
                yield Button(
                    "New session",
                    id="new-session",
                    compact=True,
                    flat=True,
                )
                yield Button(
                    "Resume",
                    variant="success",
                    id="resume-session",
                    compact=True,
                    flat=True,
                )

    def on_mount(self) -> None:
        """填充本地时间、ID、来源和事件数，并聚焦表格。"""
        table = self.query_one("#session-table", DataTable)
        table.add_columns("Updated", "Session", "Source", "Events")
        for summary in self.summaries:
            updated = summary.last_event_at or summary.created_at
            updated_text = (
                updated.astimezone().strftime("%Y-%m-%d %H:%M") if updated else "-"
            )
            table.add_row(
                Text(updated_text),
                Text(summary.session_id),
                Text(str(summary.source or "-")),
                Text(str(summary.event_count)),
                key=summary.session_id,
            )
        table.focus()

    @on(DataTable.RowSelected, "#session-table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """双击/回车行时用 row key 对应 Session ID 关闭 Screen。"""
        self.dismiss(str(event.row_key.value))

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """New 返回 None；Resume 返回当前 cursor 行 Session ID。"""
        if event.button.id == "new-session":
            self.dismiss(None)
            return
        table = self.query_one("#session-table", DataTable)
        self.dismiss(self.summaries[table.cursor_row].session_id)

    def action_new_session(self) -> None:
        """Escape action：不恢复任何 Session。"""
        self.dismiss(None)


class TraceScreen(ModalScreen[None]):
    """浏览当前 Session Trace metrics、事件行与选中 payload 的 Modal。"""

    BINDINGS = [Binding("escape", "close", show=False)]

    def __init__(self, report: dict[str, Any]) -> None:
        """建立 seq 字符串到事件的索引，并初始化未选择状态。"""
        super().__init__()
        self.report = report
        self.events_by_seq = {
            str(event["seq"]): event for event in report.get("events", [])
        }
        self.selected_event_seq: str | None = None

    def compose(self) -> ComposeResult:
        """声明 metrics、事件 DataTable、payload Syntax 和 Close action。"""
        with Vertical(id="trace-dialog"):
            with Horizontal(id="trace-heading"):
                yield Label("Trace", id="trace-title")
                yield Button(
                    "Close",
                    id="close-trace",
                    compact=True,
                    flat=True,
                )
            yield Static(
                _trace_metrics(self.report, dark=self.app.current_theme.dark),
                id="trace-metrics",
            )
            yield DataTable(
                id="trace-events",
                cursor_type="row",
                zebra_stripes=True,
            )
            yield Label("Event payload", id="trace-payload-title")
            yield Static(id="trace-payload")

    def on_mount(self) -> None:
        """填充事件表，默认选中最近事件并显示 payload。"""
        table = self.query_one("#trace-events", DataTable)
        table.add_columns("Seq", "Time", "Event", "Turn", "Summary")
        for event in self.report.get("events", []):
            timestamp = str(event.get("timestamp") or "-")
            table.add_row(
                Text(str(event.get("seq") or "-")),
                Text(timestamp[11:19] if len(timestamp) >= 19 else timestamp),
                Text(str(event.get("type") or "-")),
                Text(str(event.get("turn_id") or "-")),
                Text(str(event.get("summary") or "")),
                key=str(event["seq"]),
            )
        if table.row_count:
            table.move_cursor(row=table.row_count - 1)
            self._update_payload(str(self.report["events"][-1]["seq"]))
            table.focus()

    @on(DataTable.RowHighlighted, "#trace-events")
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """光标移动时实时更新 payload。"""
        self._update_payload(str(event.row_key.value))

    @on(DataTable.RowSelected, "#trace-events")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """行激活时更新 payload。"""
        self._update_payload(str(event.row_key.value))

    @on(Button.Pressed, "#close-trace")
    def on_close_pressed(self) -> None:
        """Close button 回调：关闭 Modal。"""
        self.dismiss(None)

    def action_close(self) -> None:
        """Escape action：关闭 Modal。"""
        self.dismiss(None)

    def _update_payload(self, seq: str) -> None:
        """以排序 JSON/Syntax 显示选中事件，最多保留 20,000 字符。"""
        event = self.events_by_seq.get(seq)
        if event is None:
            return
        self.selected_event_seq = seq
        detail = {
            "event_id": event.get("event_id"),
            "turn_id": event.get("turn_id"),
            "timestamp": event.get("timestamp"),
            "payload": event.get("payload", {}),
        }
        serialized = json.dumps(detail, ensure_ascii=False, indent=2, sort_keys=True)
        if len(serialized) > 20_000:
            serialized = serialized[:20_000] + "\n... truncated"
        self.query_one("#trace-payload", Static).update(
            Syntax(
                serialized,
                "json",
                theme="ansi_dark" if self.app.current_theme.dark else "ansi_light",
                word_wrap=True,
                background_color="default",
            )
        )


def _trace_metrics(report: dict[str, Any], *, dark: bool) -> Table:
    """按当前 palette 渲染事件/Turn/Tool/失败/审批/终态两行摘要。"""
    palette = palette_for(dark)
    metrics = report.get("metrics", {})
    table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(6):
        table.add_column(ratio=1)
    table.add_row(
        Text("events", style=palette.muted),
        str(metrics.get("event_count", 0)),
        Text("turns", style=palette.muted),
        str(metrics.get("turn_count", 0)),
        Text("tools", style=palette.muted),
        str(metrics.get("tool_call_count", 0)),
    )
    table.add_row(
        Text("failures", style=palette.muted),
        str(metrics.get("tool_failure_count", 0)),
        Text("approvals", style=palette.muted),
        str(metrics.get("approval_count", 0)),
        Text("status", style=palette.muted),
        str(metrics.get("final_status", "unknown")),
    )
    return table
