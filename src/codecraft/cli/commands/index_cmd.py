from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.retrieval import RepositoryIndex


def register_index_command(app: typer.Typer) -> None:
    """注册同步 workspace 索引并打印增量统计的 ``index`` 子命令。"""

    @app.command("index")
    def index_command(
        path: Annotated[
            Path,
            typer.Argument(help="Workspace directory to index."),
        ] = Path("."),
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        max_file_bytes: Annotated[
            int,
            typer.Option(
                "--max-file-bytes",
                min=1,
                help="Skip files larger than this limit.",
            ),
        ] = 1_000_000,
    ) -> None:
        """验证目录，在线程中增量同步本地 SQLite 索引并打印可诊断统计。

        ``codecraft index`` 是可选的显式预热，不是运行 Agent 的前置条件；没有数据库时
        workspace_search 会降级为实时扫描。同步属于阻塞的目录遍历、解析和 SQLite I/O，
        因而放进工作线程，避免以后从已有事件循环复用命令逻辑时阻塞异步任务。
        """
        workspace = path.expanduser().resolve()
        console = make_console()
        if not workspace.is_dir():
            console.print(
                f"Workspace is not a directory: {workspace}",
                style="error",
                markup=False,
            )
            raise typer.Exit(code=2)
        index = RepositoryIndex(codecraft_home.expanduser().resolve() / "indexes")
        try:
            stats = asyncio.run(
                asyncio.to_thread(
                    index.sync,
                    workspace,
                    max_file_bytes=max_file_bytes,
                )
            )
        except (OSError, ValueError) as exc:
            console.print(f"Indexing failed: {exc}", style="error", markup=False)
            raise typer.Exit(code=1) from exc

        console.print(f"index_workspace: {workspace}", style="muted", soft_wrap=True)
        console.print(
            f"index_files: indexed={stats.indexed_file_count} "
            f"updated={stats.updated_file_count} "
            f"unchanged={stats.unchanged_file_count} "
            f"deleted={stats.deleted_file_count}",
            style="muted",
            soft_wrap=True,
        )
        console.print(
            f"index_content: chunks={stats.chunk_count} symbols={stats.symbol_count} "
            f"bytes={stats.indexed_bytes}",
            style="muted",
            soft_wrap=True,
        )
        console.print(
            f"index_database: {stats.database_path}", style="muted", soft_wrap=True
        )
