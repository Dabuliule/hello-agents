from __future__ import annotations

import asyncio
import json
from threading import Event
from types import SimpleNamespace
from typing import cast

from codecraft.core.event_bus import EventBus
from codecraft.core.session import Session
from codecraft.core.session_store import SessionStore
from codecraft.core.thread import AgentThread
from codecraft.llm import MockProvider
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ToolRegistry
from codecraft.approval.policy import ApprovalPolicy
from codecraft.sandbox.policy import SandboxMode


def _config(tmp_path) -> SessionConfig:
    return SessionConfig(
        session_id="ses_io",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
    )


def _event(*, seq: int) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"evt_{seq}",
        session_id="ses_io",
        seq=seq,
        type=RuntimeEventType.TURN_STARTED,
    )


def test_session_store_runs_append_io_outside_the_event_loop(tmp_path, monkeypatch):
    async def run_test() -> None:
        store = SessionStore(tmp_path / ".codecraft")
        await store.create_session(_config(tmp_path))
        entered = Event()
        release = Event()
        original = store._append_line

        def blocked_append(path, line: str) -> None:
            entered.set()
            if not release.wait(timeout=1):
                raise AssertionError("append blocked the event loop")
            original(path, line)

        monkeypatch.setattr(store, "_append_line", blocked_append)
        append = asyncio.create_task(store.append_event(_event(seq=1)))

        while not entered.is_set():
            await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(append, timeout=1)

        assert [event.seq for event in await store.load_events("ses_io")] == [1]

    asyncio.run(run_test())


def test_session_store_serializes_concurrent_appends(tmp_path):
    async def run_test() -> None:
        store = SessionStore(tmp_path / ".codecraft")
        await store.create_session(_config(tmp_path))
        events = [
            _event(seq=index).model_copy(update={"payload": {"content": "x" * 200_000}})
            for index in range(1, 65)
        ]

        await asyncio.gather(*(store.append_event(event) for event in events))

        lines = await store.load_raw_lines("ses_io")
        records = [json.loads(line) for line in lines]
        assert len(records) == len(events)
        assert {record["event_id"] for record in records} == {
            event.event_id for event in events
        }
        assert [event.seq for event in await store.load_events("ses_io")] == list(
            range(1, 65)
        )

    asyncio.run(run_test())


def test_session_emit_broadcasts_a_persisted_event_before_cancellation(
    tmp_path, monkeypatch
):
    async def run_test() -> None:
        config = _config(tmp_path)
        store = SessionStore(config.codecraft_home)
        await store.create_session(config)
        event_bus = EventBus()
        broadcast: list[RuntimeEvent] = []

        async def capture(event: RuntimeEvent) -> None:
            broadcast.append(event)

        event_bus.subscribe(capture)
        session = Session(
            config=config,
            session_store=store,
            llm_provider=MockProvider([]),
            tool_registry=ToolRegistry(),
            event_bus=event_bus,
        )
        entered = Event()
        release = Event()
        original = store._append_line

        def blocked_append(path, line: str) -> None:
            entered.set()
            if not release.wait(timeout=1):
                raise AssertionError("append did not receive its release signal")
            original(path, line)

        monkeypatch.setattr(store, "_append_line", blocked_append)
        emit = asyncio.create_task(session.emit(RuntimeEventType.TURN_STARTED))

        while not entered.is_set():
            await asyncio.sleep(0)
        emit.cancel()
        await asyncio.sleep(0)
        assert not emit.done()

        release.set()
        try:
            await emit
        except asyncio.CancelledError:
            pass

        persisted = await store.load_events(config.session_id)
        assert [event.seq for event in persisted] == [1]
        assert [event.seq for event in broadcast] == [1]

    asyncio.run(run_test())


def test_session_store_finishes_append_before_propagating_cancellation(
    tmp_path, monkeypatch
):
    async def run_test() -> None:
        store = SessionStore(tmp_path / ".codecraft")
        await store.create_session(_config(tmp_path))
        entered = Event()
        release = Event()
        original = store._append_line

        def blocked_append(path, line: str) -> None:
            entered.set()
            if not release.wait(timeout=1):
                raise AssertionError("append did not receive its release signal")
            original(path, line)

        monkeypatch.setattr(store, "_append_line", blocked_append)
        append = asyncio.create_task(store.append_event(_event(seq=1)))

        while not entered.is_set():
            await asyncio.sleep(0)
        append.cancel()
        await asyncio.sleep(0)
        assert not append.done()

        release.set()
        try:
            await append
        except asyncio.CancelledError:
            pass

        assert len(await store.load_raw_lines("ses_io")) == 1

    asyncio.run(run_test())


def test_session_store_finishes_creation_before_propagating_cancellation(
    tmp_path, monkeypatch
):
    async def run_test() -> None:
        store = SessionStore(tmp_path / ".codecraft")
        entered = Event()
        release = Event()
        original = store._create_session_file

        def blocked_create(path) -> None:
            entered.set()
            if not release.wait(timeout=1):
                raise AssertionError("creation did not receive its release signal")
            original(path)

        monkeypatch.setattr(store, "_create_session_file", blocked_create)
        creation = asyncio.create_task(store.create_session(_config(tmp_path)))

        while not entered.is_set():
            await asyncio.sleep(0)
        creation.cancel()
        await asyncio.sleep(0)
        assert not creation.done()

        release.set()
        try:
            await creation
        except asyncio.CancelledError:
            pass

        assert await store.load_raw_lines("ses_io") == []

    asyncio.run(run_test())


def test_agent_thread_does_not_block_producers_without_a_consumer():
    session = cast(Session, SimpleNamespace(event_bus=EventBus()))
    thread = AgentThread(session)

    async def run_test() -> None:
        for seq in range(1, 2_049):
            await session.event_bus.emit(_event(seq=seq))
        assert thread._events.qsize() == 2_048

    asyncio.run(run_test())
