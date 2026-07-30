from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codecraft.schema.event import EventPayload


class ErrorRenderer:
    def __init__(self, console: Console) -> None:
        self.console = console

    def render_error(self, payload: EventPayload) -> None:
        message = payload.get("message") or payload.get("error") or "Runtime failed."
        code = payload.get("code")
        suggestion = payload.get("suggestion")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        if code:
            table.add_row("code", str(code))
        table.add_row("message", str(message))
        if suggestion:
            table.add_row("suggestion", str(suggestion))
        self.console.print(Panel(table, title="error", border_style="error"))

    def render_aborted(self, payload: EventPayload) -> None:
        message = payload.get("message") or "Turn aborted."
        self.console.print(Panel(str(message), title="aborted", border_style="warning"))
