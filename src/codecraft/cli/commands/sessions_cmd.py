from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.cli.ui.session_renderer import SessionRenderer
from codecraft.core.session_store import SessionStore


def register_sessions_command(app: typer.Typer) -> None:
    """注册列出有效或包含损坏日志的 ``sessions`` 子命令。"""

    @app.command("sessions")
    def sessions_command(
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        all_sessions: Annotated[
            bool,
            typer.Option("--all", help="Include invalid session logs."),
        ] = False,
    ) -> None:
        """同步桥接异步 SessionStore 列表查询。"""
        import asyncio

        asyncio.run(
            run_sessions(codecraft_home=codecraft_home, all_sessions=all_sessions)
        )


async def run_sessions(*, codecraft_home: Path, all_sessions: bool) -> None:
    """加载 Session summaries，并用统一 Renderer 输出或提示空列表。"""
    summaries = await SessionStore(codecraft_home).list_sessions(
        include_invalid=all_sessions
    )
    console = make_console()
    if not summaries:
        console.print("No sessions found.")
        return

    SessionRenderer(console).render_sessions(summaries)
