from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from codecraft.cli.ui.approval_renderer import ApprovalRenderer
from codecraft.cli.ui.error_renderer import ErrorRenderer
from codecraft.cli.ui.render_config import RenderConfig
from codecraft.cli.ui.tool_renderer import ToolRenderer
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput


class RuntimeEventRenderer:
    def __init__(
        self,
        *,
        console: Console,
        render_config: RenderConfig | None = None,
        tool_renderer: ToolRenderer | None = None,
        approval_renderer: ApprovalRenderer | None = None,
        error_renderer: ErrorRenderer | None = None,
    ) -> None:
        self.console = console
        self.render_config = render_config or RenderConfig()
        self.tool_renderer = tool_renderer or ToolRenderer(console, self.render_config)
        self.approval_renderer = approval_renderer or ApprovalRenderer(console)
        self.error_renderer = error_renderer or ErrorRenderer(console)
        self._streaming = False
        self._stream_buffer: list[str] = []
        self._stream_live: Live | None = None
        self._event_handlers: dict[RuntimeEventType, Callable[[RuntimeEvent], None]] = {
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA: self._render_assistant_delta,
            RuntimeEventType.ASSISTANT_MESSAGE: self._render_assistant_message,
            RuntimeEventType.TOOL_CALL_STARTED: self._render_tool_started,
            RuntimeEventType.TOOL_CALL_FINISHED: self._render_tool_finished,
            RuntimeEventType.APPROVAL_DECIDED: self._render_approval_decided,
            RuntimeEventType.PATCH_APPLIED: self._render_patch_applied,
            RuntimeEventType.TOKEN_COUNT: self._render_token_count,
            RuntimeEventType.CONTEXT_COMPACTED: self._render_context_compacted,
            RuntimeEventType.ERROR: self._render_error,
            RuntimeEventType.TURN_ABORTED: self._render_turn_aborted,
            RuntimeEventType.SESSION_RESTORED: self._render_session_restored,
            RuntimeEventType.SESSION_STARTED: self._render_debug_event,
            RuntimeEventType.TURN_STARTED: self._render_debug_event,
            RuntimeEventType.USER_MESSAGE: self._render_debug_event,
            RuntimeEventType.MODEL_TOOL_CALL: self._render_debug_event,
            RuntimeEventType.TURN_FINISHED: self._render_debug_event,
            RuntimeEventType.SESSION_CLOSED: self._render_debug_event,
        }

    async def render(self, event: RuntimeEvent) -> None:
        handler = self._event_handlers.get(event.type)
        if handler is not None:
            handler(event)

    async def request_approval(self, event: RuntimeEvent) -> SessionInput:
        self.ensure_newline()
        return await self.approval_renderer.request_decision(event)

    def ensure_newline(self) -> None:
        if self._streaming:
            text = "".join(self._stream_buffer)
            self._finish_stream(text)

    def _render_assistant_delta(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text")
        if isinstance(text, str):
            self._streaming = True
            self._stream_buffer.append(text)
            self._render_stream()

    def _render_assistant_message(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text")
        if not isinstance(text, str):
            return
        if self._streaming:
            self._finish_stream(text)
        else:
            self._render_markdown(text)

    def _render_tool_started(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.tool_renderer.render_started(event.payload)

    def _render_tool_finished(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.tool_renderer.render_finished(event.payload)

    def _render_approval_decided(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        approved = event.payload.get("approved")
        style = "success" if approved else "warning"
        label = "approved" if approved else "rejected"
        self.console.print(f"[{style}]approval {label}[/{style}]")

    def _render_patch_applied(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.tool_renderer.render_patch_applied(event.payload)

    def _render_token_count(self, event: RuntimeEvent) -> None:
        if self.render_config.debug:
            self.console.print(f"[muted]tokens {event.payload}[/muted]")

    def _render_context_compacted(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.console.print("[warning]context compacted[/warning]")

    def _render_error(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.error_renderer.render_error(event.payload)

    def _render_turn_aborted(self, event: RuntimeEvent) -> None:
        self.ensure_newline()
        self.error_renderer.render_aborted(event.payload)

    def _render_session_restored(self, event: RuntimeEvent) -> None:
        if self.render_config.debug:
            self.console.print("[muted]session restored[/muted]")

    def _render_debug_event(self, event: RuntimeEvent) -> None:
        if self.render_config.debug:
            self.console.print(f"[muted]{event.type} {event.payload}[/muted]")

    def _render_stream(self) -> None:
        if not self.console.is_terminal:
            return

        text = "".join(self._stream_buffer)
        renderable = Markdown(text)
        if self._stream_live is None:
            self._stream_live = Live(
                renderable,
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._stream_live.start()
        else:
            self._stream_live.update(renderable, refresh=True)

    def _finish_stream(self, text: str) -> None:
        if self._stream_live is not None:
            self._stream_live.update(Markdown(text), refresh=True)
            self._stream_live.stop()
            self._stream_live = None
        elif text:
            self._render_markdown(text)

        self._streaming = False
        self._stream_buffer.clear()

    def _render_markdown(self, text: str) -> None:
        self.console.print(Markdown(text))
