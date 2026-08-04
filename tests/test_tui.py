from __future__ import annotations

import asyncio

from typer.testing import CliRunner

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.cli.app import app
from codecraft.cli.bootstrap import build_runtime
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.llm import (
    LLMProvider,
    LLMProviderError,
    LLMProviderRegistry,
    MockProvider,
    ModelCompletedEvent,
    ModelMessageCompletedEvent,
    ModelMessageDeltaEvent,
    ModelRequest,
    ModelTokenCountEvent,
    ModelToolCallEvent,
)
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ToolRegistry, WriteFileTool
from codecraft.tui import (
    ActivityBlock,
    CodeCraftTUI,
    MessageBlock,
    SessionBrowserScreen,
    TUIColorScheme,
    TraceScreen,
)
from codecraft.tui.commands import (
    ComposerMenuMode,
    command_choices,
    parse_composer_menu,
)

runner = CliRunner()


class RecoveringProvider(LLMProvider):
    name = "recovering"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest):
        self.calls += 1
        if self.calls == 1:
            yield ModelMessageDeltaEvent(
                payload={"text": "partial answer"},
            )
            raise LLMProviderError("transient provider failure")
        yield ModelMessageCompletedEvent(
            payload={"text": "recovered answer"},
        )
        yield ModelCompletedEvent()


def _config(tmp_path, *, approval_policy=ApprovalPolicy.NEVER) -> SessionConfig:
    return SessionConfig(
        session_id="ses_tui",
        source=SessionSource.CLI_TUI,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy=approval_policy,
        sandbox_mode="workspace_write",
    )


async def _wait_until(pilot, predicate, *, attempts=100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.01)
    raise AssertionError("TUI condition was not reached")


async def _seed_session(config: SessionConfig) -> None:
    store = SessionStore(config.codecraft_home)
    await store.create_session(config)
    events = [
        RuntimeEvent(
            event_id="evt_started",
            session_id=config.session_id,
            seq=1,
            type=RuntimeEventType.SESSION_STARTED,
            payload={"config": config.model_dump(mode="json")},
        ),
        RuntimeEvent(
            event_id="evt_user",
            session_id=config.session_id,
            turn_id="turn_one",
            seq=2,
            type=RuntimeEventType.USER_MESSAGE,
            payload={"input_id": "inp_one", "text": "first question"},
        ),
        RuntimeEvent(
            event_id="evt_tokens",
            session_id=config.session_id,
            turn_id="turn_one",
            seq=3,
            type=RuntimeEventType.TOKEN_COUNT,
            payload={
                "input_tokens": 4,
                "output_tokens": 3,
                "total_tokens": 7,
            },
        ),
        RuntimeEvent(
            event_id="evt_assistant",
            session_id=config.session_id,
            turn_id="turn_one",
            seq=4,
            type=RuntimeEventType.ASSISTANT_MESSAGE,
            payload={"text": "first answer"},
        ),
        RuntimeEvent(
            event_id="evt_finished",
            session_id=config.session_id,
            turn_id="turn_one",
            seq=5,
            type=RuntimeEventType.TURN_FINISHED,
            payload={"answer": "first answer", "tool_calls": 0, "duration_ms": 1},
        ),
    ]
    for event in events:
        await store.append_event(event)


def test_composer_menu_parses_commands_and_skill_mentions():
    slash = parse_composer_menu("/sta")
    skill_command = parse_composer_menu("/skills front")
    skill_mention = parse_composer_menu("review with $front")

    assert slash is not None
    assert slash.mode == ComposerMenuMode.COMMANDS
    assert slash.query == "sta"
    assert [choice.value for choice in command_choices(slash.query)] == ["status"]

    assert skill_command is not None
    assert skill_command.mode == ComposerMenuMode.SKILLS
    assert skill_command.query == "front"
    assert skill_command.replace_start == 0

    assert skill_mention is not None
    assert skill_mention.mode == ComposerMenuMode.SKILLS
    assert skill_mention.query == "front"
    assert skill_mention.replace_start == len("review with ")
    assert parse_composer_menu("normal message") is None


def test_tui_slash_menu_filters_commands_and_activates_selected_skill(tmp_path):
    async def run_test():
        skill_directory = tmp_path / ".codecraft" / "skills" / "frontend-review"
        skill_directory.mkdir(parents=True)
        skill_body = "SLASH_SELECTED_SKILL_BODY"
        (skill_directory / "SKILL.md").write_text(
            "\n".join(
                [
                    "---",
                    "name: frontend-review",
                    "description: Review frontend usability and accessibility.",
                    "---",
                    "",
                    skill_body,
                ]
            ),
            encoding="utf-8",
        )
        provider = MockProvider(
            [
                ModelMessageCompletedEvent(
                    payload={"text": "skill applied"},
                ),
                ModelCompletedEvent(),
            ]
        )
        config = _config(tmp_path)
        runtime = build_runtime(
            config,
            llm_providers=LLMProviderRegistry([provider]),
        )
        tui = CodeCraftTUI(config, runtime, browse_sessions=False)

        async with tui.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            prompt = tui.query_one("#prompt")
            menu = tui.query_one("#composer-menu")
            options = tui.query_one("#composer-options")

            prompt.value = "/"
            await pilot.pause()
            assert menu.display is True
            assert options.option_count == 6
            assert tui.focused is prompt
            assert menu.region.bottom <= tui.query_one("#prompt-shell").region.y

            await pilot.press("down")
            assert options.highlighted == 1
            await pilot.press("up")
            assert options.highlighted == 0

            await pilot.press("escape")
            assert menu.display is False
            assert tui.focused is prompt

            prompt.value = "/sta"
            await pilot.pause()
            assert options.option_count == 1
            assert options.get_option_at_index(0).id == "command-status"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: any(
                    block.role == "Status" for block in tui.query(MessageBlock)
                ),
            )
            assert provider.calls == []
            assert prompt.value == ""
            assert menu.display is False

            prompt.value = "/does-not-exist"
            await pilot.pause()
            assert options.highlighted is None
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: any(
                    block.role == "Error" and "Unknown command" in block.text
                    for block in tui.query(MessageBlock)
                ),
            )
            assert provider.calls == []

            prompt.value = "/status extra"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: any(
                    block.role == "Error"
                    and block.text == "Unknown command: /status extra"
                    for block in tui.query(MessageBlock)
                ),
            )
            assert provider.calls == []

            prompt.value = "/skills"
            await pilot.pause()
            assert options.option_count == 1
            assert options.get_option_at_index(0).id == "skill-frontend-review"
            await pilot.press("tab")
            assert prompt.value == "$frontend-review "
            assert menu.display is False

            prompt.value += "review this screen"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle"
                    and any(
                        block.role == "Assistant" and block.text == "skill applied"
                        for block in tui.query(MessageBlock)
                    )
                ),
            )

            assert len(provider.calls) == 1
            system = provider.calls[0].messages[0].content or ""
            assert skill_body in system
            assert system.count("## Skill: frontend-review") == 1

            snapshot = await tui.thread.read_snapshot()
            assert not any(
                event.type == RuntimeEventType.MODEL_TOOL_CALL
                for event in snapshot.events
            )

    asyncio.run(run_test())


def test_tui_composer_menu_layout_does_not_overlap_at_narrow_or_wide_sizes(
    tmp_path,
):
    async def run_test() -> None:
        appearances = (
            (TUIColorScheme.LIGHT, "codecraft-light", "#EEF0F2", "#FFFFFF"),
            (TUIColorScheme.DARK, "codecraft-dark", "#0E0F11", "#16181B"),
        )
        for width, height in ((60, 24), (120, 40)):
            for scheme, theme_name, background, surface in appearances:
                await assert_layout(
                    width,
                    height,
                    scheme,
                    theme_name,
                    background,
                    surface,
                )

    async def assert_layout(
        width: int,
        height: int,
        scheme: TUIColorScheme,
        theme_name: str,
        background: str,
        surface: str,
    ) -> None:
        config = _config(tmp_path).model_copy(
            update={"session_id": f"ses_menu_{width}_{scheme}"}
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(
            config,
            runtime,
            browse_sessions=False,
            color_scheme=scheme,
        )

        async with tui.run_test(size=(width, height)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            assert tui.theme == theme_name
            assert tui.screen.styles.background.hex == background
            assert tui.query_one("#prompt-shell").styles.background.hex == surface
            prompt = tui.query_one("#prompt")
            prompt.value = "/"
            await pilot.pause()

            conversation = tui.query_one("#conversation-pane")
            composer_frame = tui.query_one("#composer-frame")
            menu = tui.query_one("#composer-menu")
            prompt_shell = tui.query_one("#prompt-shell")
            status = tui.query_one("#runtime-status")

            assert conversation.region.bottom <= composer_frame.region.y
            assert menu.region.bottom <= prompt_shell.region.y
            assert prompt_shell.region.bottom <= status.region.y
            assert status.region.bottom <= height

    asyncio.run(run_test())


def test_tui_streams_messages_and_updates_runtime_status(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        provider = MockProvider(
            [
                ModelMessageDeltaEvent(
                    payload={"text": "Hello "},
                ),
                ModelTokenCountEvent(
                    payload={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
                ModelMessageDeltaEvent(
                    payload={"text": "world"},
                ),
                ModelCompletedEvent(),
            ]
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime)

        async with tui.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            prompt = tui.query_one("#prompt")
            prompt.value = "hello request"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle"
                    and len(list(tui.query(MessageBlock))) == 2
                ),
            )

            messages = list(tui.query(MessageBlock))
            assert [(message.role, message.text) for message in messages] == [
                ("User", "hello request"),
                ("Assistant", "Hello world"),
            ]
            assert tui.token_usage == {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            }
            assert prompt.disabled is False
            conversation = tui.query_one("#conversation-pane")
            header = tui.query_one("#session-header")
            composer = tui.query_one("#composer")
            assert header.region.bottom <= conversation.region.y
            assert conversation.region.bottom <= composer.region.y
            assert conversation.region.width == tui.screen.region.width
            assert not list(tui.query("#side-panel"))

    asyncio.run(run_test())


def test_tui_single_column_layout_fits_narrow_terminal(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime, browse_sessions=False)

        async with tui.run_test(size=(50, 18)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            await tui._append_message(
                "User",
                "Inspect the session lifecycle on a narrow terminal.",
            )
            await tui._render_tool_started(
                {
                    "call_id": "call_narrow",
                    "name": "read_file",
                    "arguments": {
                        "path": "src/codecraft/core/session.py",
                    },
                }
            )
            await pilot.pause()

            screen = tui.screen.region
            header = tui.query_one("#session-header")
            conversation = tui.query_one("#conversation-pane")
            composer = tui.query_one("#composer")
            prompt_shell = tui.query_one("#prompt-shell")
            message = list(tui.query(MessageBlock))[-1]
            activity = list(tui.query(ActivityBlock))[-1]

            for widget in (
                header,
                conversation,
                composer,
                prompt_shell,
                message,
                activity,
            ):
                assert widget.region.x >= screen.x
                assert widget.region.right <= screen.right
            assert conversation.region.bottom <= composer.region.y
            assert message.region.width == activity.region.width

    asyncio.run(run_test())


def test_tui_recovers_after_aborted_stream_without_reusing_message_block(tmp_path):
    async def run_test():
        config = _config(tmp_path).model_copy(update={"model_provider": "recovering"})
        provider = RecoveringProvider()
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime)

        async with tui.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            prompt = tui.query_one("#prompt")
            prompt.value = "first request"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle"
                    and any(
                        message.role == "Error" for message in tui.query(MessageBlock)
                    )
                ),
            )

            assert prompt.disabled is False
            prompt.value = "retry"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle"
                    and any(
                        message.text == "recovered answer"
                        for message in tui.query(MessageBlock)
                    )
                ),
            )

            messages = [
                (message.role, message.text) for message in tui.query(MessageBlock)
            ]
            assert messages == [
                ("User", "first request"),
                ("Assistant", "partial answer"),
                ("Error", "transient provider failure"),
                ("User", "retry"),
                ("Assistant", "recovered answer"),
            ]
            assert provider.calls == 2

    asyncio.run(run_test())


def test_tui_renders_tool_payloads_as_plain_text_and_disables_closed_session(
    tmp_path,
):
    async def run_test():
        config = _config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime)

        async with tui.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            await tui._render_tool_started(
                {
                    "call_id": "call_markup",
                    "name": "[red]spoof[/red] [link=https://example.test]tool[/link]",
                    "arguments": {"path": "[bold]README.md[/bold]"},
                }
            )
            await tui._render_tool_finished(
                {
                    "call_id": "call_markup",
                    "name": "[red]spoof[/red] [link=https://example.test]tool[/link]",
                    "result": {"success": True},
                }
            )
            await pilot.pause()

            activities = list(tui.query(ActivityBlock))
            assert len(activities) == 1
            assert activities[0].status == "completed"
            rendered = activities[0].render().plain
            assert "[red]spoof[/red]" in rendered
            assert "[bold]README.md[/bold]" in rendered
            assert "[link=https://example.test]tool[/link]" in rendered

            tui._accumulate_token_usage(
                {
                    "input_tokens": -1,
                    "output_tokens": True,
                    "total_tokens": 4,
                }
            )
            assert tui.token_usage == {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 4,
            }

            await tui._handle_event(
                RuntimeEvent(
                    event_id="evt_closed",
                    session_id=config.session_id,
                    seq=999,
                    type=RuntimeEventType.SESSION_CLOSED,
                )
            )
            assert tui.turn_status == "closed"
            assert tui.query_one("#prompt").disabled is True

    asyncio.run(run_test())


def test_tui_inline_approval_controls_side_effect(tmp_path):
    async def run_test():
        config = _config(tmp_path, approval_policy=ApprovalPolicy.ON_REQUEST)
        provider = MockProvider(
            [
                ModelToolCallEvent(
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {
                            "path": "approved.txt",
                            "content": "approved by TUI\n",
                        },
                    },
                ),
                ModelCompletedEvent(),
                ModelMessageCompletedEvent(
                    payload={"text": "File created."},
                ),
                ModelCompletedEvent(),
            ]
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([WriteFileTool()]),
            approval_manager=ApprovalManager(
                reviewer=ThreadApprovalReviewer(),
            ),
        )
        tui = CodeCraftTUI(config, runtime)

        async with tui.run_test(size=(120, 40)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            prompt = tui.query_one("#prompt")
            prompt.value = "create a file"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "approval"
                    and tui.query_one("#approval-prompt").display
                ),
            )

            assert not (tmp_path / "approved.txt").exists()
            options = tui.query_one("#approval-options")
            assert tui.focused is options
            assert options.highlighted == 0
            await pilot.press("down")
            assert options.highlighted == 1
            await pilot.press("enter")

            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle" and (tmp_path / "approved.txt").exists()
                ),
            )
            assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == (
                "approved by TUI\n"
            )
            assert list(tui.query(MessageBlock))[-1].text == "File created."
            assert tui.query_one("#approval-prompt").display is False
            assert tui.query_one("#prompt-shell").display is True

    asyncio.run(run_test())


def test_tui_inline_approval_defaults_to_reject_and_escape_closes_it(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime, browse_sessions=False)

        async with tui.run_test(size=(60, 24)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            decision = asyncio.create_task(
                tui._show_inline_approval(
                    {
                        "tool_name": "write_file",
                        "risk": "writes workspace files",
                        "reason": "The tool will modify a file.",
                    }
                )
            )
            await _wait_until(
                pilot,
                lambda: tui.query_one("#approval-prompt").display,
            )

            options = tui.query_one("#approval-options")
            assert options.highlighted == 0
            await pilot.press("escape")
            assert await decision is False
            assert tui.query_one("#approval-prompt").display is False
            assert tui.query_one("#prompt-shell").display is True

    asyncio.run(run_test())


def test_tui_trace_screen_inspects_persisted_events(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        provider = MockProvider(
            [
                ModelMessageCompletedEvent(
                    payload={"text": "trace answer"},
                ),
                ModelCompletedEvent(),
            ]
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime)

        async with tui.run_test(size=(80, 24)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")
            prompt = tui.query_one("#prompt")
            prompt.value = "trace this"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: tui.turn_status == "idle")

            await pilot.press("ctrl+t")
            await _wait_until(pilot, lambda: isinstance(tui.screen, TraceScreen))
            trace_screen = tui.screen
            report = trace_screen.report
            assert report["metrics"]["turn_count"] == 1
            assert report["metrics"]["error_count"] == 0

            table = trace_screen.query_one("#trace-events")
            assert table.row_count == report["metrics"]["event_count"]
            user_row = next(
                index
                for index, event in enumerate(report["events"])
                if event["type"] == RuntimeEventType.USER_MESSAGE
            )
            table.move_cursor(row=user_row)
            await pilot.pause()
            user_seq = str(report["events"][user_row]["seq"])
            assert trace_screen.selected_event_seq == user_seq
            assert trace_screen.events_by_seq[user_seq]["payload"]["text"] == (
                "trace this"
            )

            metrics = trace_screen.query_one("#trace-metrics")
            payload_title = trace_screen.query_one("#trace-payload-title")
            assert metrics.region.bottom <= table.region.y
            assert table.region.bottom <= payload_title.region.y
            assert await pilot.click("#close-trace") is True
            await _wait_until(pilot, lambda: not isinstance(tui.screen, TraceScreen))
            assert prompt.disabled is False

    asyncio.run(run_test())


def test_tui_browses_resumes_and_continues_session(tmp_path):
    class CountingSessionStore(SessionStore):
        def __init__(self, codecraft_home) -> None:
            super().__init__(codecraft_home)
            self.resume_calls = 0

        async def resume(self, session_id: str):
            self.resume_calls += 1
            return await super().resume(session_id)

    async def run_test():
        startup_config = _config(tmp_path).model_copy(update={"session_id": "ses_new"})
        stored_config = _config(tmp_path).model_copy(
            update={"session_id": "ses_stored"}
        )
        await _seed_session(stored_config)

        provider = MockProvider(
            [
                ModelMessageCompletedEvent(
                    payload={"text": "continued answer"},
                ),
                ModelCompletedEvent(),
            ]
        )
        stores: list[CountingSessionStore] = []

        def build_runtime(config: SessionConfig) -> AgentRuntime:
            store = CountingSessionStore(config.codecraft_home)
            stores.append(store)
            return AgentRuntime(
                session_store=store,
                llm_providers=LLMProviderRegistry([provider]),
                tool_registry=ToolRegistry(),
            )

        tui = CodeCraftTUI(
            startup_config,
            build_runtime(startup_config),
            runtime_factory=build_runtime,
        )

        async with tui.run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: (
                    isinstance(tui.screen, SessionBrowserScreen)
                    and tui.screen.query_one("#session-table").row_count == 1
                ),
            )
            table = tui.screen.query_one("#session-table")
            assert table.row_count == 1
            assert await pilot.click("#resume-session") is True
            await _wait_until(pilot, lambda: tui.turn_status == "idle")

            assert tui.config.session_id == "ses_stored"
            assert [
                (message.role, message.text) for message in tui.query(MessageBlock)
            ] == [
                ("User", "first question"),
                ("Assistant", "first answer"),
            ]
            assert tui.token_usage["total_tokens"] == 7

            prompt = tui.query_one("#prompt")
            prompt.value = "continue"
            await pilot.press("enter")
            await _wait_until(
                pilot,
                lambda: (
                    tui.turn_status == "idle"
                    and len(list(tui.query(MessageBlock))) == 4
                ),
            )
            assert list(tui.query(MessageBlock))[-1].text == "continued answer"
            assert len(stores) == 2
            assert [store.resume_calls for store in stores] == [1, 0]

    asyncio.run(run_test())


def test_tui_resume_last_uses_store_fallback_without_listing_first(tmp_path):
    class CountingSessionStore(SessionStore):
        def __init__(self, codecraft_home) -> None:
            super().__init__(codecraft_home)
            self.resume_last_calls = 0

        async def resume_last(self, cwd=None):
            self.resume_last_calls += 1
            return await super().resume_last(cwd=cwd)

    class CountingRuntime(AgentRuntime):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.list_calls = 0

        async def list_sessions(self, cwd=None):
            self.list_calls += 1
            return await super().list_sessions(cwd=cwd)

    async def run_test() -> None:
        config = _config(tmp_path)
        await _seed_session(config)
        store = CountingSessionStore(config.codecraft_home)
        runtime = CountingRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        tui = CodeCraftTUI(config, runtime, resume_last=True)

        async with tui.run_test(size=(100, 32)) as pilot:
            await _wait_until(pilot, lambda: tui.turn_status == "idle")

            assert store.resume_last_calls == 1
            assert runtime.list_calls == 0
            assert [
                (message.role, message.text) for message in tui.query(MessageBlock)
            ] == [
                ("User", "first question"),
                ("Assistant", "first answer"),
            ]

    asyncio.run(run_test())


def test_root_command_uses_tui_session_source(tmp_path, monkeypatch):
    captured: list[SessionConfig] = []

    def fake_run(self) -> None:
        captured.append(self.config)

    monkeypatch.setattr(CodeCraftTUI, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert captured[0].source == SessionSource.CLI_TUI
    assert captured[0].model_provider == "mock"


def test_root_command_forwards_direct_resume_options(tmp_path, monkeypatch):
    captured: list[CodeCraftTUI] = []

    def fake_run(self) -> None:
        captured.append(self)

    monkeypatch.setattr(CodeCraftTUI, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
            "--resume",
            "ses_existing",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].resume_session_id == "ses_existing"
    assert captured[0].resume_last is False
