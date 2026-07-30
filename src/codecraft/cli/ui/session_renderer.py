from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionSummary


class SessionRenderer:
    def __init__(self, console: Console) -> None:
        self.console = console

    def render_sessions(self, summaries: Iterable[SessionSummary]) -> None:
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
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        table.add_row("session", session_id)
        table.add_row("events", str(len(events)))
        table.add_row("last event", str(events[-1].type) if events else "-")
        table.add_row("final answer", "available" if last_answer(events) else "-")
        self.console.print(Panel(table, title="session inspect", border_style="cyan"))

    def render_events(self, events: list[RuntimeEvent]) -> None:
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
        return "valid" if summary.valid else f"invalid:{summary.error_code or '-'}"


def shorten_path(path: Path | None, *, max_chars: int = 48) -> str:
    if path is None:
        return "-"
    text = str(path)
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars + 3 :]


def last_answer(events: list[RuntimeEvent]) -> str | None:
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_FINISHED:
            answer = event.payload.get("answer")
            if isinstance(answer, str):
                return answer
    return None


def event_summary(event: RuntimeEvent) -> str:
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
