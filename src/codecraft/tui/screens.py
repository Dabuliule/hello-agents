from __future__ import annotations

import json
from typing import Any

from rich.console import Group
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


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "reject", show=False)]

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__()
        self.payload = payload

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("Approval required", id="approval-title")
            yield Static(_approval_details(self.payload), id="approval-details")
            with Horizontal(id="approval-actions"):
                yield Button("Reject", variant="error", id="reject")
                yield Button("Approve", variant="success", id="approve")

    def on_mount(self) -> None:
        self.query_one("#reject", Button).focus()

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_reject(self) -> None:
        self.dismiss(False)


class SessionBrowserScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "new_session", show=False)]

    def __init__(self, summaries: list[SessionSummary]) -> None:
        super().__init__()
        self.summaries = summaries

    def compose(self) -> ComposeResult:
        with Vertical(id="session-dialog"):
            yield Label("Sessions", id="session-title")
            yield DataTable(
                id="session-table",
                cursor_type="row",
                zebra_stripes=True,
            )
            with Horizontal(id="session-actions"):
                yield Button("New session", id="new-session")
                yield Button("Resume", variant="success", id="resume-session")

    def on_mount(self) -> None:
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
        self.dismiss(str(event.row_key.value))

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-session":
            self.dismiss(None)
            return
        table = self.query_one("#session-table", DataTable)
        self.dismiss(self.summaries[table.cursor_row].session_id)

    def action_new_session(self) -> None:
        self.dismiss(None)


class TraceScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", show=False)]

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__()
        self.report = report
        self.events_by_seq = {
            str(event["seq"]): event for event in report.get("events", [])
        }
        self.selected_event_seq: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="trace-dialog"):
            with Horizontal(id="trace-heading"):
                yield Label("Trace", id="trace-title")
                yield Button("Close", id="close-trace")
            yield Static(_trace_metrics(self.report), id="trace-metrics")
            yield DataTable(
                id="trace-events",
                cursor_type="row",
                zebra_stripes=True,
            )
            yield Label("Event payload", id="trace-payload-title")
            yield Static(id="trace-payload")

    def on_mount(self) -> None:
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
        self._update_payload(str(event.row_key.value))

    @on(DataTable.RowSelected, "#trace-events")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self._update_payload(str(event.row_key.value))

    @on(Button.Pressed, "#close-trace")
    def on_close_pressed(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

    def _update_payload(self, seq: str) -> None:
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
                theme="ansi_dark",
                word_wrap=True,
                background_color="default",
            )
        )


def _trace_metrics(report: dict[str, Any]) -> Table:
    metrics = report.get("metrics", {})
    table = Table.grid(padding=(0, 1), expand=True)
    for _ in range(6):
        table.add_column(ratio=1)
    table.add_row(
        Text("events", style="#8f98a3"),
        str(metrics.get("event_count", 0)),
        Text("turns", style="#8f98a3"),
        str(metrics.get("turn_count", 0)),
        Text("tools", style="#8f98a3"),
        str(metrics.get("tool_call_count", 0)),
    )
    table.add_row(
        Text("failures", style="#8f98a3"),
        str(metrics.get("tool_failure_count", 0)),
        Text("approvals", style="#8f98a3"),
        str(metrics.get("approval_count", 0)),
        Text("status", style="#8f98a3"),
        str(metrics.get("final_status", "unknown")),
    )
    return table


def _approval_details(payload: dict[str, Any]) -> Group:
    table = Table.grid(padding=(0, 2), expand=True)
    table.add_column(style="#9ca3ad", width=10)
    table.add_column(ratio=1)
    table.add_row("tool", Text(str(payload.get("tool_name") or "-")))
    table.add_row("risk", Text(str(payload.get("risk") or "-")))
    table.add_row("reason", Text(str(payload.get("reason") or "-")))
    arguments = payload.get("arguments")
    details = ""
    if isinstance(arguments, dict) and arguments:
        details = json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True)
        if len(details) > 2_000:
            details = details[:2_000] + "\n[truncated]"
    return Group(table, Text(details, style="#d9dde3"))
