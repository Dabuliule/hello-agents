from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from codecraft.cli.runtime_runner import shutdown_thread
from codecraft.core.errors import CodecraftError
from codecraft.core.ids import new_id
from codecraft.core.runtime import AgentRuntime
from codecraft.core.thread import AgentThread
from codecraft.core.trace_report import build_trace_report
from codecraft.schema.event import EventPayload, RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig
from codecraft.tui.commands import (
    ComposerChoiceKind,
    ComposerMenuMode,
    parse_composer_menu,
)
from codecraft.tui.composer import ComposerMenu
from codecraft.tui.screens import SessionBrowserScreen, TraceScreen
from codecraft.tui.theme import (
    CODECRAFT_THEMES,
    CODECRAFT_THEME_VARIABLE_DEFAULTS,
    TUIColorScheme,
    palette_for,
    textual_theme_name,
)
from codecraft.tui.widgets import (
    ActivityBlock,
    MessageBlock,
    RuntimeStatusLine,
    SessionHeader,
)


MAX_RESTORED_MESSAGES = 100
MAX_RESTORED_TOOL_EVENTS = 200


class CodeCraftTUI(App[None]):
    TITLE = "CodeCraft"
    BINDINGS = [
        Binding("ctrl+q", "quit", show=False, priority=True),
        Binding("ctrl+c", "quit", show=False, priority=True),
        Binding("ctrl+t", "trace", show=False),
        Binding("escape", "reject_approval", show=False),
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
        color_scheme: TUIColorScheme = TUIColorScheme.LIGHT,
    ) -> None:
        super().__init__()
        for theme in CODECRAFT_THEMES:
            self.register_theme(theme)
        self.theme = textual_theme_name(color_scheme)
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
        self._activity_blocks: dict[str, ActivityBlock] = {}
        self._approval_result: asyncio.Future[bool] | None = None
        self._last_error_turn_id: str | None = None
        self._closed = False
        self._runtime_event_handlers: dict[
            RuntimeEventType,
            Callable[[RuntimeEvent], Awaitable[None]],
        ] = {
            RuntimeEventType.TURN_STARTED: self._handle_turn_started,
            RuntimeEventType.USER_MESSAGE: self._handle_user_message,
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA: self._handle_assistant_delta,
            RuntimeEventType.ASSISTANT_MESSAGE: self._handle_assistant_message,
            RuntimeEventType.TOOL_CALL_STARTED: self._handle_tool_started,
            RuntimeEventType.TOOL_CALL_FINISHED: self._handle_tool_finished,
            RuntimeEventType.APPROVAL_REQUESTED: self._handle_approval_requested,
            RuntimeEventType.TOKEN_COUNT: self._handle_token_count,
            RuntimeEventType.CONTEXT_COMPACTED: self._handle_context_compacted,
            RuntimeEventType.SESSION_RESTORED: self._handle_session_restored,
            RuntimeEventType.ERROR: self._handle_runtime_error,
            RuntimeEventType.TURN_ABORTED: self._handle_turn_aborted,
            RuntimeEventType.TURN_FINISHED: self._handle_turn_finished,
            RuntimeEventType.SESSION_CLOSED: self._handle_session_closed,
        }

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return {
            **super().get_theme_variable_defaults(),
            **CODECRAFT_THEME_VARIABLE_DEFAULTS,
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="app-shell"):
            with Center(id="header-frame"):
                yield SessionHeader(self.config, id="session-header")
            yield VerticalScroll(id="conversation-pane")
            with Center(id="composer-frame"):
                with Vertical(id="composer"):
                    yield ComposerMenu(id="composer-menu")
                    with Vertical(id="approval-prompt"):
                        yield Static(id="approval-inline-title")
                        yield Static(id="approval-inline-detail")
                        yield OptionList(
                            Option("Reject", id="reject"),
                            Option("Approve once", id="approve"),
                            id="approval-options",
                            markup=False,
                            compact=True,
                        )
                    with Horizontal(id="prompt-shell"):
                        yield Static("›", id="prompt-prefix")
                        yield Input(
                            placeholder="Ask CodeCraft",
                            id="prompt",
                            disabled=True,
                        )
                    yield RuntimeStatusLine(self.config, id="runtime-status")

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
            await self._append_activity(
                ActivityBlock.notice(f"{omitted_tools} earlier tool results omitted")
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
                await self._render_tool_finished(event.payload)
            elif event.type == RuntimeEventType.TOKEN_COUNT:
                self._accumulate_token_usage(event.payload)
        self._refresh_status()

    @on(Input.Submitted, "#prompt")
    async def on_prompt_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.thread is None or self.turn_status != "idle":
            return
        menu_query = parse_composer_menu(event.value)
        if menu_query is not None:
            if await self._accept_composer_choice():
                return
            event.input.value = ""
            self._close_composer_menu()
            message = (
                "No matching skills were found."
                if menu_query.mode == ComposerMenuMode.SKILLS
                else f"Unknown command: {text}"
            )
            await self._append_message("Error", message)
            event.input.focus()
            return
        if text.startswith("/"):
            event.input.value = ""
            self._close_composer_menu()
            await self._append_message("Error", f"Unknown command: {text}")
            event.input.focus()
            return
        event.input.value = ""
        self._close_composer_menu()
        event.input.disabled = True
        self.turn_status = "running"
        self._refresh_status()
        try:
            await self.thread.submit(SessionInput.user_message(new_id("inp_"), text))
        except Exception as exc:
            await self._append_message("Error", f"Could not submit message: {exc}")
            self._finish_turn("failed")

    @on(Input.Changed, "#prompt")
    def on_prompt_changed(self, event: Input.Changed) -> None:
        if self.turn_status != "idle" or event.input.disabled:
            self._close_composer_menu()
            return
        self._refresh_composer_menu(event.value)

    async def on_key(self, event: events.Key) -> None:
        if not self._composer_menu_open() or self.focused is not self.query_one(
            "#prompt", Input
        ):
            return
        if event.key == "down":
            self.query_one(ComposerMenu).move(1)
        elif event.key == "up":
            self.query_one(ComposerMenu).move(-1)
        elif event.key == "tab":
            await self._accept_composer_choice()
        else:
            return
        event.prevent_default()
        event.stop()

    @on(OptionList.OptionSelected, "#composer-options")
    async def on_composer_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        if event.option.id is not None:
            await self._accept_composer_choice(event.option.id)

    async def action_trace(self) -> None:
        try:
            events = await self.runtime.session_store.load_events(
                self.config.session_id
            )
        except CodecraftError as exc:
            await self._append_activity(
                ActivityBlock.notice(
                    f"Trace unavailable: {exc.message}",
                    failed=True,
                )
            )
            return
        report = build_trace_report(self.config.session_id, events)
        self.push_screen(TraceScreen(report))

    def action_reject_approval(self) -> None:
        if self._composer_menu_open():
            self._close_composer_menu()
            self.query_one("#prompt", Input).focus()
            return
        if self._approval_result is not None and not self._approval_result.done():
            self._approval_result.set_result(False)

    def _refresh_composer_menu(self, value: str) -> None:
        opened = self.query_one(ComposerMenu).refresh_for(
            value,
            self.runtime.skill_registry.list(),
        )
        self.query_one("#composer-frame").set_class(opened, "composer-menu-active")
        self.query_one("#composer").set_class(opened, "composer-menu-active")

    def _close_composer_menu(self) -> None:
        self.query_one(ComposerMenu).close()
        self.query_one("#composer-frame").remove_class("composer-menu-active")
        self.query_one("#composer").remove_class("composer-menu-active")

    def _composer_menu_open(self) -> bool:
        return self.query_one("#composer-menu").display

    async def _accept_composer_choice(self, choice_id: str | None = None) -> bool:
        menu = self.query_one(ComposerMenu)
        choice = menu.selected_choice(choice_id)
        if choice is None:
            return False

        prompt = self.query_one("#prompt", Input)
        if choice.kind == ComposerChoiceKind.SKILL:
            updated = menu.insert_skill(prompt.value, choice.value)
            if updated is None:
                return False
            prompt.value = updated
            prompt.cursor_position = len(prompt.value)
            self._close_composer_menu()
            prompt.focus()
            return True

        prompt.value = ""
        self._close_composer_menu()
        if choice.value == "skills":
            prompt.value = "/skills "
            prompt.cursor_position = len(prompt.value)
            prompt.focus()
            return True
        await self._execute_slash_command(choice.value)
        if not self._closed and choice.value != "trace":
            prompt.focus()
        return True

    async def _execute_slash_command(self, command: str) -> None:
        if command == "status":
            await self._append_message(
                "Status",
                "\n".join(
                    [
                        f"session: {self.config.session_id}",
                        f"model: {self.config.model_provider}/{self.config.model}",
                        f"approval: {self.config.approval_policy}",
                        f"sandbox: {self.config.sandbox_mode}",
                        f"skills: {len(self.runtime.skill_registry.list())}",
                    ]
                ),
            )
        elif command == "tools":
            tools = self.runtime.tool_registry.specs()
            await self._append_message(
                "Tools",
                "\n".join(spec.name for spec in tools) or "No tools available.",
            )
        elif command == "mcp":
            servers = [
                f"{name}: {'enabled' if settings.enabled else 'disabled'}"
                for name, settings in self.config.mcp_servers.items()
            ]
            await self._append_message(
                "MCP",
                "\n".join(servers) or "No MCP servers configured.",
            )
        elif command == "trace":
            await self.action_trace()
        elif command == "quit":
            await self.action_quit()

    @on(OptionList.OptionSelected, "#approval-options")
    def on_approval_selected(self, event: OptionList.OptionSelected) -> None:
        if self._approval_result is None or self._approval_result.done():
            return
        self._approval_result.set_result(event.option.id == "approve")

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
        handler = self._runtime_event_handlers.get(event.type)
        if handler is not None:
            await handler(event)

    async def _handle_turn_started(self, event: RuntimeEvent) -> None:
        self._last_error_turn_id = None
        self.turn_status = "running"
        self._close_composer_menu()
        self.query_one("#prompt", Input).disabled = True
        self._refresh_status()

    async def _handle_user_message(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text")
        if isinstance(text, str):
            await self._append_message("User", text)

    async def _handle_assistant_delta(self, event: RuntimeEvent) -> None:
        delta = event.payload.get("text")
        if not isinstance(delta, str):
            return
        self._assistant_buffer += delta
        if self._assistant_block is None:
            self._assistant_block = await self._append_message("Assistant", "")
        self._assistant_block.set_text(self._assistant_buffer)
        self._scroll_conversation()

    async def _handle_assistant_message(self, event: RuntimeEvent) -> None:
        text = event.payload.get("text")
        if not isinstance(text, str):
            return
        if self._assistant_block is None:
            self._assistant_block = await self._append_message("Assistant", text)
        else:
            self._assistant_block.set_text(text)
        self._assistant_block = None
        self._assistant_buffer = ""
        self._scroll_conversation()

    async def _handle_tool_started(self, event: RuntimeEvent) -> None:
        await self._render_tool_started(event.payload)

    async def _handle_tool_finished(self, event: RuntimeEvent) -> None:
        await self._render_tool_finished(event.payload)

    async def _handle_approval_requested(self, event: RuntimeEvent) -> None:
        await self._request_approval(event.payload)

    async def _handle_token_count(self, event: RuntimeEvent) -> None:
        self._add_token_usage(event.payload)

    async def _handle_context_compacted(self, event: RuntimeEvent) -> None:
        await self._append_activity(ActivityBlock.notice("Context compacted"))

    async def _handle_session_restored(self, event: RuntimeEvent) -> None:
        await self._append_activity(ActivityBlock.notice("Session restored"))

    async def _handle_runtime_error(self, event: RuntimeEvent) -> None:
        payload = event.payload
        message = str(payload.get("message") or payload.get("code") or "Error")
        if event.turn_id is None or event.turn_id != self._last_error_turn_id:
            await self._append_message("Error", message)
        self._last_error_turn_id = event.turn_id

    async def _handle_turn_aborted(self, event: RuntimeEvent) -> None:
        payload = event.payload
        message = str(payload.get("message") or payload.get("reason") or "Aborted")
        if event.turn_id is None or event.turn_id != self._last_error_turn_id:
            await self._append_message("Error", message)
        self._last_error_turn_id = None
        self._finish_turn("idle")

    async def _handle_turn_finished(self, event: RuntimeEvent) -> None:
        self._last_error_turn_id = None
        self._finish_turn("idle")

    async def _handle_session_closed(self, event: RuntimeEvent) -> None:
        self._last_error_turn_id = None
        self._finish_turn("closed")

    async def _request_approval(self, payload: EventPayload) -> None:
        if self.thread is None:
            return
        activity = self._activity_for_payload(payload)
        if activity is not None:
            activity.mark_waiting()
        self.turn_status = "approval"
        self._refresh_status()
        approved = await self._show_inline_approval(payload)
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
        if activity is not None:
            activity.mark_running()
        self.turn_status = "running"
        self._refresh_status()

    async def _show_inline_approval(self, payload: EventPayload) -> bool:
        if self._approval_result is not None:
            raise RuntimeError("another approval decision is already active")

        palette = palette_for(self.current_theme.dark)
        title = Text("Approval required", style=f"bold {palette.strong}")
        tool_name = str(payload.get("tool_name") or "tool")
        title.append("  ·  ", style=palette.separator)
        title.append(tool_name, style=palette.warning)
        reason = str(payload.get("reason") or payload.get("risk") or "")

        self.query_one("#approval-inline-title", Static).update(title)
        self.query_one("#approval-inline-detail", Static).update(Text(reason))
        self.query_one("#prompt-shell").display = False
        self.query_one("#approval-prompt").display = True
        self.query_one("#composer-frame").add_class("approval-active")
        self.query_one("#composer").add_class("approval-active")

        options = self.query_one("#approval-options", OptionList)
        options.highlighted = 0
        options.focus()
        self._approval_result = asyncio.get_running_loop().create_future()
        try:
            return await self._approval_result
        finally:
            self._approval_result = None
            self.query_one("#approval-prompt").display = False
            self.query_one("#prompt-shell").display = True
            self.query_one("#composer-frame").remove_class("approval-active")
            self.query_one("#composer").remove_class("approval-active")

    async def _append_message(self, role: str, text: str) -> MessageBlock:
        block = MessageBlock(role, text)
        await self.query_one("#conversation-pane", VerticalScroll).mount(block)
        self._scroll_conversation()
        return block

    def _scroll_conversation(self) -> None:
        self.query_one("#conversation-pane", VerticalScroll).scroll_end(animate=False)

    async def _append_activity(self, block: ActivityBlock) -> ActivityBlock:
        conversation = self.query_one("#conversation-pane", VerticalScroll)
        if not conversation.children or not isinstance(
            conversation.children[-1], ActivityBlock
        ):
            block.add_class("activity-group-start")
        await conversation.mount(block)
        self._scroll_conversation()
        return block

    async def _render_tool_started(self, payload: EventPayload) -> None:
        call_id_value = payload.get("call_id")
        call_id = call_id_value if isinstance(call_id_value, str) else None
        name = str(payload.get("name") or "tool")
        arguments = payload.get("arguments")
        block = await self._append_activity(
            ActivityBlock(
                name,
                call_id=call_id,
                arguments=arguments if isinstance(arguments, dict) else None,
            )
        )
        if call_id is not None:
            self._activity_blocks[call_id] = block

    async def _render_tool_finished(self, payload: EventPayload) -> None:
        call_id_value = payload.get("call_id")
        call_id = call_id_value if isinstance(call_id_value, str) else None
        name = str(payload.get("name") or "tool")
        block = self._activity_blocks.pop(call_id, None) if call_id else None
        if block is None:
            block = self._latest_running_activity(name)
        if block is None:
            block = await self._append_activity(ActivityBlock(name, call_id=call_id))
        block.finish(payload)
        self._scroll_conversation()

    def _activity_for_payload(self, payload: EventPayload) -> ActivityBlock | None:
        call_id = payload.get("call_id")
        if not isinstance(call_id, str):
            return None
        return self._activity_blocks.get(call_id)

    def _latest_running_activity(self, name: str) -> ActivityBlock | None:
        blocks = list(self.query(ActivityBlock))
        for block in reversed(blocks):
            if block.tool_name == name and block.status in {"running", "waiting"}:
                if block.call_id is not None:
                    self._activity_blocks.pop(block.call_id, None)
                return block
        return None

    def _add_token_usage(self, payload: EventPayload) -> None:
        self._accumulate_token_usage(payload)
        self._refresh_status()

    def _accumulate_token_usage(self, payload: EventPayload) -> None:
        for name in self.token_usage:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.token_usage[name] += value

    def _finish_turn(self, status: str) -> None:
        self._assistant_block = None
        self._assistant_buffer = ""
        for block in self._activity_blocks.values():
            block.mark_stopped()
        self._activity_blocks.clear()
        self.turn_status = status
        self._close_composer_menu()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = self._closed or status != "idle"
        if status == "idle":
            prompt.focus()
        self._refresh_status()

    def _refresh_status(self) -> None:
        self.query_one("#session-header", SessionHeader).set_config(self.config)
        self.query_one("#runtime-status", RuntimeStatusLine).set_state(
            self.config,
            self.turn_status,
            self.token_usage,
        )

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
