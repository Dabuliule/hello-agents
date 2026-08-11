from __future__ import annotations

from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from codecraft.core.ids import new_id
from codecraft.schema.event import EventPayload, RuntimeEvent
from codecraft.schema.input import SessionInput

AskFn = Callable[[str], Awaitable[str]]


class ApprovalRenderer:
    """把 APPROVAL_REQUESTED 渲染为 Rich 面板并交互生成 SessionInput。"""

    def __init__(self, console: Console, ask: AskFn | None = None) -> None:
        """绑定 Console 和可注入的异步输入函数，便于测试。"""
        self.console = console
        self.ask = ask or self._default_ask

    async def request_decision(self, event: RuntimeEvent) -> SessionInput:
        """循环接受 y/n/d；EOF/中断 fail-closed 为拒绝。"""
        payload = event.payload
        self.render_request(payload)
        while True:
            try:
                answer = (await self.ask("Approve? [y/n/d] ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer in {"y", "yes"}:
                return self._decision(payload, approved=True, reason="approved by CLI")
            if answer in {"n", "no", ""}:
                return self._decision(payload, approved=False, reason="rejected by CLI")
            if answer == "d":
                self.render_details(payload)

    def render_request(self, payload: EventPayload) -> None:
        """展示工具、风险、原因及 Bash command/cwd 的最小审批摘要。"""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="muted")
        table.add_column()
        table.add_row("tool", str(payload.get("tool_name") or "-"))
        table.add_row("risk", str(payload.get("risk") or "-"))
        table.add_row("reason", str(payload.get("reason") or "-"))
        arguments = payload.get("arguments")
        if isinstance(arguments, dict):
            command = arguments.get("command")
            cwd = arguments.get("cwd")
            if command:
                table.add_row("command", str(command))
            if cwd:
                table.add_row("cwd", str(cwd))
        self.console.print(
            Panel(table, title="approval required", border_style="approval")
        )

    def render_details(self, payload: EventPayload) -> None:
        """按用户 d 请求展示完整类型化 payload。"""
        self.console.print(
            Panel(str(payload), title="approval details", border_style="approval")
        )

    def _decision(
        self, payload: EventPayload, *, approved: bool, reason: str
    ) -> SessionInput:
        """使用事件 approval_id 构造用户 Reviewer 可消费的旁路决定输入。"""
        return SessionInput.approval_decision(
            new_id("inp_"),
            approval_id=str(payload["approval_id"]),
            approved=approved,
            reason=reason,
        )

    async def _default_ask(self, prompt: str) -> str:
        """默认使用终端 input；接口保持 async 以兼容事件消费循环。"""
        return input(prompt)
