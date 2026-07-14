from __future__ import annotations

import asyncio

from typer.testing import CliRunner

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.cli.app import app
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.llm import (
    LLMProvider,
    LLMProviderError,
    LLMProviderRegistry,
    MockProvider,
    ModelEvent,
    ModelEventType,
    ModelRequest,
)
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ToolRegistry, WriteFileTool
from codecraft.tui import (
    ApprovalScreen,
    CodeCraftTUI,
    MessageBlock,
    SessionBrowserScreen,
    TraceScreen,
)

runner = CliRunner()


class RecoveringProvider(LLMProvider):
    name = "recovering"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest):
        self.calls += 1
        if self.calls == 1:
            yield ModelEvent(
                type=ModelEventType.MESSAGE_DELTA,
                payload={"text": "partial answer"},
            )
            raise LLMProviderError("transient provider failure")
        yield ModelEvent(
            type=ModelEventType.MESSAGE_COMPLETED,
            payload={"text": "recovered answer"},
        )
        yield ModelEvent(type=ModelEventType.COMPLETED)


def _config(tmp_path, *, approval_policy=ApprovalPolicy.NEVER) -> SessionConfig:
    return SessionConfig(
        session_id="ses_tui",
        source=SessionSource.CLI_TUI,
        cwd=tmp_path,
        workspace_roots=[tmp_path],
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
            payload={"text": "first question"},
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
        ),
    ]
    for event in events:
        await store.append_event(event)


def test_tui_streams_messages_and_updates_runtime_status(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "Hello "},
                ),
                ModelEvent(
                    type=ModelEventType.TOKEN_COUNT,
                    payload={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "world"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
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
            side_panel = tui.query_one("#side-panel")
            main = tui.query_one("#main")
            assert conversation.region.right <= side_panel.region.x
            assert main.region.bottom <= prompt.region.y
            assert conversation.region.width >= 30

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
            tui._render_tool_started(
                {
                    "name": "[red]spoof[/red]",
                    "arguments": {"path": "[bold]README.md[/bold]"},
                }
            )
            tui._render_tool_finished(
                {
                    "name": "[link=https://example.test]tool[/link]",
                    "result": {"success": True},
                }
            )
            await pilot.pause()

            log = tui.query_one("#tool-log")
            rendered = "".join(line.text for line in log.lines)
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


def test_tui_approval_modal_controls_side_effect(tmp_path):
    async def run_test():
        config = _config(tmp_path, approval_policy=ApprovalPolicy.ON_REQUEST)
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {
                            "path": "approved.txt",
                            "content": "approved by TUI\n",
                        },
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "File created."},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
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
            await _wait_until(pilot, lambda: isinstance(tui.screen, ApprovalScreen))

            assert not (tmp_path / "approved.txt").exists()
            assert tui.screen.focused is not None
            assert tui.screen.focused.id == "reject"
            clicked = await pilot.click("#approve")
            assert clicked is True

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

    asyncio.run(run_test())


def test_tui_trace_screen_inspects_persisted_events(tmp_path):
    async def run_test():
        config = _config(tmp_path)
        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "trace answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
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

            assert await pilot.click("#open-trace") is True
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
    async def run_test():
        startup_config = _config(tmp_path).model_copy(update={"session_id": "ses_new"})
        stored_config = _config(tmp_path).model_copy(
            update={"session_id": "ses_stored"}
        )
        await _seed_session(stored_config)

        provider = MockProvider(
            [
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "continued answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )

        def build_runtime(config: SessionConfig) -> AgentRuntime:
            return AgentRuntime(
                session_store=SessionStore(config.codecraft_home),
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
                pilot, lambda: isinstance(tui.screen, SessionBrowserScreen)
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
