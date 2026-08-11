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
    """按 RuntimeEventType 分发 Rich 渲染，并聚合 Assistant 流式 Markdown。"""

    def __init__(
        self,
        *,
        console: Console,
        render_config: RenderConfig | None = None,
        tool_renderer: ToolRenderer | None = None,
        approval_renderer: ApprovalRenderer | None = None,
        error_renderer: ErrorRenderer | None = None,
    ) -> None:
        """装配子 Renderer、流状态和覆盖全部事件类型的 handler 表。"""
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
        """调用事件类型对应同步 handler；无 handler 时安全忽略。"""
        handler = self._event_handlers.get(event.type)
        if handler is not None:
            handler(event)

    async def request_approval(self, event: RuntimeEvent) -> SessionInput:
        """先结束流式 Assistant 区域，再交互获取审批输入。"""
        self.ensure_newline()
        return await self.approval_renderer.request_decision(event)

    def ensure_newline(self) -> None:
        """若正流式输出，将累计文本收口为最终 Markdown 块。"""
        if self._streaming:
            text = "".join(self._stream_buffer)
            self._finish_stream(text)

    def _render_assistant_delta(self, event: RuntimeEvent) -> None:
        """累计合法文本 delta，并在真实终端刷新 Live Markdown。"""
        text = event.payload.get("text")
        if isinstance(text, str):
            self._streaming = True
            self._stream_buffer.append(text)
            self._render_stream()

    def _render_assistant_message(self, event: RuntimeEvent) -> None:
        """流式时用完整消息收口校准；非流式直接渲染 Markdown。"""
        text = event.payload.get("text")
        if not isinstance(text, str):
            return
        if self._streaming:
            self._finish_stream(text)
        else:
            self._render_markdown(text)

    def _render_tool_started(self, event: RuntimeEvent) -> None:
        """收口文本流并渲染工具开始摘要。"""
        self.ensure_newline()
        self.tool_renderer.render_started(event.payload)

    def _render_tool_finished(self, event: RuntimeEvent) -> None:
        """收口文本流并渲染工具终态。"""
        self.ensure_newline()
        self.tool_renderer.render_finished(event.payload)

    def _render_approval_decided(self, event: RuntimeEvent) -> None:
        """显示批准/拒绝的紧凑状态行。"""
        self.ensure_newline()
        approved = event.payload.get("approved")
        style = "success" if approved else "warning"
        label = "approved" if approved else "rejected"
        self.console.print(f"[{style}]approval {label}[/{style}]")

    def _render_patch_applied(self, event: RuntimeEvent) -> None:
        """渲染 patch modified/added/deleted 审计计数。"""
        self.ensure_newline()
        self.tool_renderer.render_patch_applied(event.payload)

    def _render_token_count(self, event: RuntimeEvent) -> None:
        """只在 debug 模式打印 Token payload。"""
        if self.render_config.debug:
            self.console.print(f"[muted]tokens {event.payload}[/muted]")

    def _render_context_compacted(self, event: RuntimeEvent) -> None:
        """收口流并提示发生上下文压缩。"""
        self.ensure_newline()
        self.console.print("[warning]context compacted[/warning]")

    def _render_error(self, event: RuntimeEvent) -> None:
        """收口流并委托 ErrorRenderer 显示 Runtime 错误。"""
        self.ensure_newline()
        self.error_renderer.render_error(event.payload)

    def _render_turn_aborted(self, event: RuntimeEvent) -> None:
        """收口流并显示中止面板。"""
        self.ensure_newline()
        self.error_renderer.render_aborted(event.payload)

    def _render_session_restored(self, event: RuntimeEvent) -> None:
        """debug 模式提示 Session 已恢复。"""
        if self.render_config.debug:
            self.console.print("[muted]session restored[/muted]")

    def _render_debug_event(self, event: RuntimeEvent) -> None:
        """debug 模式原样展示低层事件类型与 payload。"""
        if self.render_config.debug:
            self.console.print(f"[muted]{event.type} {event.payload}[/muted]")

    def _render_stream(self) -> None:
        """终端环境启动/更新 Rich Live；非 TTY 等最终完整消息再输出。"""
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
        """停止 Live 或输出完整文本，并清除全部流状态。"""
        if self._stream_live is not None:
            self._stream_live.update(Markdown(text), refresh=True)
            self._stream_live.stop()
            self._stream_live = None
        elif text:
            self._render_markdown(text)

        self._streaming = False
        self._stream_buffer.clear()

    def _render_markdown(self, text: str) -> None:
        """将 Assistant 完整文本按 Rich Markdown 输出。"""
        self.console.print(Markdown(text))
