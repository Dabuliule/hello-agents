from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli.options import CodecraftHomeOption
from codecraft.schema.session import SessionSource
from codecraft.tui import CodeCraftTUI, TUIThemeMode, resolve_color_scheme


def register_root_command(app: typer.Typer) -> None:
    """注册无子命令时启动交互 TUI 的根 callback。"""

    @app.callback(invoke_without_command=True)
    def root_command(
        context: typer.Context,
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
        resume: Annotated[
            str | None,
            typer.Option("--resume", help="Resume a session by id."),
        ] = None,
        last: Annotated[
            bool,
            typer.Option(
                "--last",
                help="Resume the latest session for the current working directory.",
            ),
        ] = False,
        theme: Annotated[
            TUIThemeMode,
            typer.Option(
                "--theme",
                envvar="CODECRAFT_THEME",
                help="TUI theme: auto, light, or dark.",
            ),
        ] = TUIThemeMode.AUTO,
    ) -> None:
        """验证 resume 选项、装配 CLI_TUI Runtime 并以解析后的主题运行 Textual。

        Typer 已选择子命令时立即返回，避免根 callback 重复启动 TUI。
        """
        if context.invoked_subcommand is not None:
            return

        from codecraft.cli import app as cli_app

        if resume is not None and last:
            raise typer.BadParameter("Use either --resume or --last, not both.")

        session_config = cli_app._load_session_config(
            source=SessionSource.CLI_TUI,
            provider=provider,
            model=model,
            codecraft_home=codecraft_home,
            config_path=config,
            profile=profile,
            approval_policy=approval_policy,
            network=network,
        )
        CodeCraftTUI(
            session_config,
            cli_app._build_runtime(session_config),
            runtime_factory=cli_app._build_runtime,
            resume_session_id=resume,
            resume_last=last,
            color_scheme=resolve_color_scheme(theme),
        ).run()
