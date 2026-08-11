from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli.commands.common import build_event_renderer, render_startup_error
from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.runtime_runner import submit_user_message
from codecraft.core.errors import CodecraftError
from codecraft.schema.session import SessionSource


def register_exec_command(app: typer.Typer) -> None:
    """向 Typer 应用注册一次性 ``exec`` Agent 任务。"""

    @app.command("exec")
    def exec_command(
        task: Annotated[str, typer.Argument(help="User task to submit to Codecraft.")],
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider", help="Model provider: openai, qwen, or deepseek."
            ),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Model name."),
        ] = None,
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        config: Annotated[
            Path | None,
            typer.Option("--config", help="Highest-priority TOML config file."),
        ] = None,
        profile: Annotated[
            str | None,
            typer.Option("--profile", help="Profile name under ~/.codecraft/profiles."),
        ] = None,
        approval_policy: Annotated[
            ApprovalPolicy | None,
            typer.Option("--approval-policy", help="Approval policy."),
        ] = None,
        network: Annotated[
            bool | None,
            typer.Option("--network/--no-network", help="Allow network commands."),
        ] = None,
        debug: Annotated[
            bool,
            typer.Option("--debug", help="Show verbose runtime events."),
        ] = False,
    ) -> None:
        """将 CLI options 转交异步 runner，并用非零结果退出 shell。"""
        import asyncio

        exit_code = asyncio.run(
            run_exec(
                task=task,
                provider=provider,
                model=model,
                codecraft_home=codecraft_home,
                config_path=config,
                profile=profile,
                approval_policy=approval_policy,
                network=network,
                debug=debug,
            )
        )
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_exec(
    *,
    task: str,
    provider: str | None,
    model: str | None,
    codecraft_home: Path,
    config_path: Path | None,
    profile: str | None,
    approval_policy: ApprovalPolicy | None,
    network: bool | None,
    debug: bool = False,
) -> int:
    """构造 CLI_EXEC Runtime、提交一条任务、渲染至终态并始终关闭资源。"""
    from codecraft.cli import app as cli_app

    config = cli_app._load_session_config(
        source=SessionSource.CLI_EXEC,
        provider=provider,
        model=model,
        codecraft_home=codecraft_home,
        config_path=config_path,
        profile=profile,
        approval_policy=approval_policy,
        network=network,
    )
    runtime = cli_app._build_runtime(config)
    try:
        thread = await runtime.create_thread(config)
        renderer = build_event_renderer(debug=debug)
        return await submit_user_message(thread, renderer, task)
    except CodecraftError as exc:
        render_startup_error(exc)
        return 1
    finally:
        await runtime.close()
