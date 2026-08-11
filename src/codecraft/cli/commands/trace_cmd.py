from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.core.errors import SessionRestoreError
from codecraft.core.session_store import SessionStore
from codecraft.core.trace_report import (
    build_trace_report,
    render_trace_html,
    render_trace_json,
)


class TraceFormat(StrEnum):
    """Trace export 的 JSON、HTML 或双格式选择。"""

    JSON = "json"
    HTML = "html"
    BOTH = "both"


def register_trace_command(app: typer.Typer) -> None:
    """注册按 Session ID 导出事件 Trace 的命令。"""

    @app.command("trace")
    def trace_command(
        session_id: Annotated[
            str,
            typer.Argument(help="Session id to export as a trace report."),
        ],
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                "-o",
                help="Directory for generated trace files. Defaults to ~/.codecraft/traces.",
            ),
        ] = None,
        format: Annotated[
            TraceFormat,
            typer.Option("--format", help="Trace output format."),
        ] = TraceFormat.BOTH,
    ) -> None:
        """同步桥接异步导出，并把恢复失败映射为退出码。"""
        import asyncio

        exit_code = asyncio.run(
            run_trace(
                session_id=session_id,
                codecraft_home=codecraft_home,
                output_dir=output_dir,
                format=format,
            )
        )
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_trace(
    *,
    session_id: str,
    codecraft_home: Path,
    output_dir: Path | None,
    format: TraceFormat,
) -> int:
    """严格加载事件、构建一次报告，并按选择写 JSON/HTML。"""
    console = make_console()
    store = SessionStore(codecraft_home)
    try:
        events = await store.load_events(session_id)
    except SessionRestoreError as exc:
        if exc.code == "session_file_not_found":
            console.print(f"No session found: {session_id}")
        else:
            console.print(
                f"Could not export trace for {session_id}: {exc.message} ({exc.code})",
                markup=False,
                soft_wrap=True,
            )
        return 1

    report = build_trace_report(session_id, events)
    destination = (output_dir or codecraft_home / "traces").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if format in {TraceFormat.JSON, TraceFormat.BOTH}:
        json_path = destination / f"{session_id}.trace.json"
        json_path.write_text(render_trace_json(report), encoding="utf-8")
        written.append(json_path)
    if format in {TraceFormat.HTML, TraceFormat.BOTH}:
        html_path = destination / f"{session_id}.trace.html"
        html_path.write_text(render_trace_html(report), encoding="utf-8")
        written.append(html_path)

    console.print(f"trace_session: {session_id}", style="muted", soft_wrap=True)
    for path in written:
        label = "trace_json" if path.suffix == ".json" else "trace_html"
        console.print(f"{label}: {path}", style="muted", soft_wrap=True)
    return 0
