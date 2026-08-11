from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.mcp.server import create_repository_mcp_server


def register_mcp_server_command(app: typer.Typer) -> None:
    """注册以 stdio 运行只读 Repository MCP Server 的命令。"""

    @app.command("mcp-server")
    def mcp_server_command(
        workspace: Annotated[
            Path,
            typer.Option(
                "--workspace",
                "-w",
                help="Repository directory exposed by the MCP server.",
            ),
        ] = Path("."),
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
    ) -> None:
        """验证 workspace 后创建 FastMCP，并让 stdout 专用于 stdio 协议。"""
        root = workspace.expanduser().resolve()
        if not root.is_dir():
            make_console(stderr=True).print(
                f"Workspace is not a directory: {root}",
                style="error",
                markup=False,
            )
            raise typer.Exit(code=2)
        server = create_repository_mcp_server(
            root,
            codecraft_home=codecraft_home.expanduser().resolve(),
        )
        server.run(transport="stdio")
