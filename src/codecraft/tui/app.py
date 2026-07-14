from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Header, Input, Label, RichLog, Static

from codecraft.cli.runtime_runner import shutdown_thread
from codecraft.core.errors import CodecraftError
from codecraft.core.ids import new_id
from codecraft.core.runtime import AgentRuntime
from codecraft.core.thread import AgentThread
from codecraft.core.trace_report import build_trace_report
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig
from codecraft.tui.rendering import runtime_status
from codecraft.tui.screens import ApprovalScreen, SessionBrowserScreen, TraceScreen
from codecraft.tui.widgets import MessageBlock


MAX_RESTORED_MESSAGES = 100
MAX_RESTORED_TOOL_EVENTS = 200


class CodeCraftTUI(App[None]):
    TITLE = "CodeCraft"
    BINDINGS = [
        Binding("ctrl+q", "quit", show=False, priority=True),
        Binding("ctrl+c", "quit", show=False, priority=True),
    ]

    CSS_PATH = "codecraft.tcss"

    def __init__(
        self,
        config: SessionConfig,
        runtime: AgentRuntime,
        *,
        runtime_factory: Callable[[SessionConfig], AgentRuntime] | None = None,
        resume_session_id: str | None = None,
        resume_last: bool = False,
        browse_sessions: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.runtime_factory = runtime_factory
        self.resume_session_id = resume_session_id
        self.resume_last = resume_last
        self.browse_sessions = browse_sessions
        self.thread: AgentThread | None = None
        self.turn_status = "starting"
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._assistant_block: MessageBlock | None = None
        self._assistant_buffer = ""
        self._last_error_turn_id: str | None = None
        self._closed = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield VerticalScroll(id="conversation-pane")
            with Vertical(id="side-panel"):
                with Horizontal(id="runtime-heading"):
                    yield Label("Runtime", classes="panel-title")
                    yield Button("Trace", id="open-trace", disabled=True)
                yield Static(id="runtime-status")
                yield Label("Tool activity", classes="panel-title")
                yield RichLog(
                    id="tool-log",
                    min_width=1,
                    wrap=True,
                    markup=True,
                    auto_scroll=True,
                )
        yield Input(placeholder="Message CodeCraft", id="prompt", disabled=True)

    async def on_mount(self) -> None:
        self.sub_title = f"{self.config.model_provider}/{self.config.model}"
        self._refresh_status()
        self.run_worker(
            self._start_runtime(),
            group="runtime-startup",
            exclusive=True,
            name="runtime-startup",
        )

    async def _start_runtime(self) -> None:
        try:
            session_id = await self._select_session()
            if session_id is None:
                self.thread = await self.runtime.create_thread(self.config)
            else:
                await self._resume_session(session_id)
        except CodecraftError as exc:
            await self._show_startup_error(exc.message, exc.suggestion)
            return
        except Exception as exc:
            await self._show_startup_error(
                "Runtime could not start.", f"{type(exc).__name__}: {exc}"
            )
            return

        self.turn_status = "idle"
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self.query_one("#open-trace", Button).disabled = False
        self._refresh_status()
        self.run_worker(
            self._consume_events(),
            group="runtime-events",
            exclusive=True,
            name="runtime-events",
        )

    async def _select_session(self) -> str | None:
        if self.resume_session_id is not None:
            return self.resume_session_id

        summaries = await self.runtime.list_sessions(cwd=self.config.cwd)
        if self.resume_last:
            if not summaries:
                raise RuntimeError(
                    "No session found for the current working directory."
                )
            return summaries[0].session_id
        if not self.browse_sessions or not summaries:
            return None
        return await self.push_screen_wait(SessionBrowserScreen(summaries))

    async def _resume_session(self, session_id: str) -> None:
        snapshot = await self.runtime.session_store.resume(session_id)
        if self.runtime_factory is not None:
            previous_runtime = self.runtime
            self.runtime = self.runtime_factory(snapshot.config)
            await previous_runtime.close()
        elif snapshot.config != self.config:
            raise RuntimeError(
                "Resuming a session with different configuration requires a runtime factory."
            )

        self.config = snapshot.config
        self.sub_title = f"{self.config.model_provider}/{self.config.model}"
        await self._restore_history(snapshot.events)
        self.thread = await self.runtime.resume_thread(session_id)

    async def _restore_history(self, events: list[RuntimeEvent]) -> None:
        message_events = [
            event
            for event in events
            if event.type
            in {RuntimeEventType.USER_MESSAGE, RuntimeEventType.ASSISTANT_MESSAGE}
        ]
        visible_messages = message_events[-MAX_RESTORED_MESSAGES:]
        visible_message_seqs = {event.seq for event in visible_messages}
        omitted_messages = len(message_events) - len(visible_messages)
        if omitted_messages:
            await self._append_message(
                "History",
                f"{omitted_messages} earlier messages are hidden from this view.",
            )

        tool_events = [
            event
            for event in events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        ]
        visible_tool_seqs = {
            event.seq for event in tool_events[-MAX_RESTORED_TOOL_EVENTS:]
        }
        omitted_tools = len(tool_events) - len(visible_tool_seqs)
        if omitted_tools:
            self._tool_log().write(
                f"[dim]{omitted_tools} earlier tool results omitted[/dim]"
            )

        for event in events:
            if event.seq in visible_message_seqs:
                text = event.payload.get("text")
                if isinstance(text, str):
                    role = (
                        "User"
                        if event.type == RuntimeEventType.USER_MESSAGE
                        else "Assistant"
                    )
                    await self._append_message(role, text)
            elif event.seq in visible_tool_seqs:
                self._render_tool_finished(event.payload)
            elif event.type == RuntimeEventType.TOKEN_COUNT:
                self._accumulate_token_usage(event.payload)
        self._refresh_status()

    @on(Input.Submitted, "#prompt")
    async def on_prompt_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.thread is None or self.turn_status != "idle":
            return
        event.input.value = ""
        event.input.disabled = True
        self.turn_status = "running"
        self._refresh_status()
        try:
            await self.thread.submit(SessionInput.user_message(new_id("inp_"), text))
        except Exception as exc:
            await self._append_message("Error", f"Could not submit message: {exc}")
            self._finish_turn("failed")

    @on(Button.Pressed, "#open-trace")
    async def on_trace_pressed(self) -> None:
        try:
            events = await self.runtime.session_store.load_events(
                self.config.session_id
            )
        except CodecraftError as exc:
            self._tool_log().write(
                Text(f"trace unavailable: {exc.message}", style="red")
            )
            return
        report = build_trace_report(self.config.session_id, events)
        self.push_screen(TraceScreen(report))

    async def action_quit(self) -> None:
        await self._shutdown_runtime()
        self.exit()

    async def on_unmount(self) -> None:
        await self._shutdown_runtime()

    async def _consume_events(self) -> None:
        if self.thread is None:
            return
        try:
            while True:
                event = await self.thread.next_event()
                await self._handle_event(event)
                if event.type == RuntimeEventType.SESSION_CLOSED:
                    return
        except Exception as exc:
            if self._closed:
                return
            await self._append_message(
                "Error",
                f"Runtime event stream failed: {type(exc).__name__}: {exc}",
            )
            self._finish_turn("failed")

    async def _handle_event(self, event: RuntimeEvent) -> None:
        payload = event.payload
        if event.type == RuntimeEventType.TURN_STARTED:
            self._last_error_turn_id = None
            self.turn_status = "running"
            self.query_one("#prompt", Input).disabled = True
            self._refresh_status()
        elif event.type == RuntimeEventType.USER_MESSAGE:
            text = payload.get("text")
            if isinstance(text, str):
                await self._append_message("User", text)
        elif event.type == RuntimeEventType.ASSISTANT_MESSAGE_DELTA:
            delta = payload.get("text")
            if isinstance(delta, str):
                self._assistant_buffer += delta
                if self._assistant_block is None:
                    self._assistant_block = await self._append_message("Assistant", "")
                self._assistant_block.set_text(self._assistant_buffer)
                self._scroll_conversation()
        elif event.type == RuntimeEventType.ASSISTANT_MESSAGE:
            text = payload.get("text")
            if isinstance(text, str):
                if self._assistant_block is None:
                    self._assistant_block = await self._append_message(
                        "Assistant", text
                    )
                else:
                    self._assistant_block.set_text(text)
                self._assistant_block = None
                self._assistant_buffer = ""
                self._scroll_conversation()
        elif event.type == RuntimeEventType.TOOL_CALL_STARTED:
            self._render_tool_started(payload)
        elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
            self._render_tool_finished(payload)
        elif event.type == RuntimeEventType.APPROVAL_REQUESTED:
            await self._request_approval(payload)
        elif event.type == RuntimeEventType.TOKEN_COUNT:
            self._add_token_usage(payload)
        elif event.type == RuntimeEventType.CONTEXT_COMPACTED:
            self._tool_log().write("[yellow]context compacted[/yellow]")
        elif event.type == RuntimeEventType.SESSION_RESTORED:
            self._tool_log().write("[cyan]session restored[/cyan]")
        elif event.type == RuntimeEventType.ERROR:
            message = str(payload.get("message") or payload.get("code") or "Error")
            if event.turn_id is None or event.turn_id != self._last_error_turn_id:
                await self._append_message("Error", message)
            self._last_error_turn_id = event.turn_id
        elif event.type == RuntimeEventType.TURN_ABORTED:
            message = str(payload.get("message") or payload.get("reason") or "Aborted")
            if event.turn_id is None or event.turn_id != self._last_error_turn_id:
                await self._append_message("Error", message)
            self._last_error_turn_id = None
            self._finish_turn("idle")
        elif event.type == RuntimeEventType.TURN_FINISHED:
            self._last_error_turn_id = None
            self._finish_turn("idle")
        elif event.type == RuntimeEventType.SESSION_CLOSED:
            self._last_error_turn_id = None
            self._finish_turn("closed")

    async def _request_approval(self, payload: dict[str, Any]) -> None:
        if self.thread is None:
            return
        self.turn_status = "approval"
        self._refresh_status()
        approved = await self.push_screen_wait(ApprovalScreen(payload))
        try:
            approval_id = payload.get("approval_id")
            if not isinstance(approval_id, str) or not approval_id:
                raise ValueError("approval request is missing approval_id")
            decision = SessionInput.approval_decision(
                new_id("inp_"),
                approval_id=approval_id,
                approved=approved,
                reason="approved by TUI" if approved else "rejected by TUI",
            )
            await self.thread.submit(decision)
        except Exception as exc:
            await self._append_message(
                "Error",
                f"Could not submit approval decision: {type(exc).__name__}: {exc}",
            )
            try:
                await self.thread.interrupt("approval_submission_failed")
            except Exception:
                self._finish_turn("failed")
            return
        label = "approved" if approved else "rejected"
        style = "green" if approved else "red"
        self._tool_log().write(f"[{style}]approval {label}[/{style}]")
        self.turn_status = "running"
        self._refresh_status()

    async def _append_message(self, role: str, text: str) -> MessageBlock:
        block = MessageBlock(role, text)
        await self.query_one("#conversation-pane", VerticalScroll).mount(block)
        self._scroll_conversation()
        return block

    def _scroll_conversation(self) -> None:
        self.query_one("#conversation-pane", VerticalScroll).scroll_end(animate=False)

    def _render_tool_started(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "tool")
        arguments = payload.get("arguments")
        suffix = ""
        if isinstance(arguments, dict) and arguments:
            compact = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
            suffix = f" {compact[:180]}"
        line = Text()
        line.append("[running]", style="yellow")
        line.append(f" {name}{suffix}")
        self._tool_log().write(line)

    def _render_tool_finished(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("name") or "tool")
        duration = payload.get("duration_ms")
        result = payload.get("result")
        success = isinstance(result, dict) and result.get("success") is True
        status = "ok" if success else "failed"
        style = "green" if success else "red"
        elapsed = f" {duration}ms" if isinstance(duration, int) else ""
        line = Text()
        line.append(f"[{status}]", style=style)
        line.append(f" {name}{elapsed}")
        self._tool_log().write(line)
        if not success and isinstance(result, dict):
            content = str(result.get("content") or result.get("error") or "")
            if content:
                self._tool_log().write(Text(content[:500], style="#ef9a9a"))

    def _add_token_usage(self, payload: dict[str, Any]) -> None:
        self._accumulate_token_usage(payload)
        self._refresh_status()

    def _accumulate_token_usage(self, payload: dict[str, Any]) -> None:
        for name in self.token_usage:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.token_usage[name] += value

    def _finish_turn(self, status: str) -> None:
        self._assistant_block = None
        self._assistant_buffer = ""
        self.turn_status = status
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = self._closed or status != "idle"
        if status == "idle":
            prompt.focus()
        self._refresh_status()

    def _refresh_status(self) -> None:
        status = self.query_one("#runtime-status", Static)
        status.update(runtime_status(self.config, self.turn_status, self.token_usage))

    def _tool_log(self) -> RichLog:
        return self.query_one("#tool-log", RichLog)

    async def _show_startup_error(self, message: str, suggestion: str | None) -> None:
        self.turn_status = "failed"
        self._refresh_status()
        await self._append_message(
            "Error", message if not suggestion else f"{message}\n{suggestion}"
        )

    async def _shutdown_runtime(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.thread is not None:
            try:
                await shutdown_thread(self.thread)
            except Exception:
                pass
        try:
            await self.runtime.close()
        except Exception:
            pass
