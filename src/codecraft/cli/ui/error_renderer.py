from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codecraft.schema.event import EventPayload


class ErrorRenderer:
    """渲染 Runtime ERROR 与 TURN_ABORTED 的一致终端面板。"""

    def __init__(self, console: Console) -> None:
        """绑定输出 Console。"""
        self.console = console

    def render_error(self, payload: EventPayload) -> None:
        """显示稳定 code、message 和 suggestion，缺字段时使用安全回退。"""
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
        """以 warning 面板显示 Turn 中止原因。"""
        message = payload.get("message") or "Turn aborted."
        self.console.print(Panel(str(message), title="aborted", border_style="warning"))
