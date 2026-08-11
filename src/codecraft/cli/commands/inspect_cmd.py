from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.cli.ui.session_renderer import SessionRenderer, last_answer
from codecraft.core.errors import SessionRestoreError
from codecraft.core.session_store import SessionStore


def register_inspect_command(app: typer.Typer) -> None:
    """注册查看 Session 摘要、事件、工具、错误或 raw JSONL 的命令。"""

    @app.command("inspect")
    def inspect_command(
        session_id: Annotated[str, typer.Argument(help="Session id to inspect.")],
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        events: Annotated[
            bool,
            typer.Option("--events", help="Print every event."),
        ] = False,
        tools: Annotated[
            bool,
            typer.Option("--tools", help="Print tool call summaries."),
        ] = False,
        errors: Annotated[
            bool,
            typer.Option("--errors", help="Print error and aborted events."),
        ] = False,
        raw: Annotated[
            bool,
            typer.Option("--raw", help="Print raw JSONL lines without validation."),
        ] = False,
    ) -> None:
        """同步桥接 run_inspect，并把读取失败映射为退出码。"""
        import asyncio

        exit_code = asyncio.run(
            run_inspect(
                session_id=session_id,
                codecraft_home=codecraft_home,
                events=events,
                tools=tools,
                errors=errors,
                raw=raw,
            )
        )
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_inspect(
    *,
    session_id: str,
    codecraft_home: Path,
    events: bool,
    tools: bool,
    errors: bool,
    raw: bool,
) -> int:
    """raw 模式不验证逐行输出；普通模式严格加载后按选择渲染视图。"""
    console = make_console()
    store = SessionStore(codecraft_home)
    if raw:
        try:
            lines = await store.load_raw_lines(session_id)
        except SessionRestoreError as exc:
            print_restore_error(console, session_id, exc)
            return 1
        console.print(f"session_id: {session_id}", soft_wrap=True)
        console.print(f"raw_lines: {len(lines)}", soft_wrap=True)
        for line_number, line in enumerate(lines, start=1):
            console.print(f"{line_number}: {line}", soft_wrap=True, markup=False)
        return 0

    try:
        loaded = await store.load_events(session_id)
    except SessionRestoreError as exc:
        print_restore_error(console, session_id, exc)
        return 1
    renderer = SessionRenderer(console)
    renderer.render_inspect_summary(session_id, loaded)

    answer = last_answer(loaded)
    if answer:
        console.print(answer)

    if events:
        renderer.render_events(loaded)
    if tools:
        renderer.render_tool_events(loaded)
    if errors:
        renderer.render_error_events(loaded)
    return 0


def print_restore_error(
    console: Console,
    session_id: str,
    exc: SessionRestoreError,
) -> None:
    """区分日志不存在与损坏/版本等恢复错误，输出人类可读诊断。"""
    if exc.code == "session_file_not_found":
        console.print(f"No session found: {session_id}")
        return
    console.print(
        f"Could not inspect session {session_id}: {exc.message} ({exc.code})",
        markup=False,
        soft_wrap=True,
    )
