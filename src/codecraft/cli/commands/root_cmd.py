from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli.options import CodecraftHomeOption
from codecraft.schema.session import SessionSource
from codecraft.tui import CodeCraftTUI, TUIThemeMode, resolve_color_scheme


def register_root_command(app: typer.Typer) -> None:
    """注册无子命令时启动交互 TUI 的根 callback。

    Args:
        app: 已创建的 CodeCraft Typer 应用。

    根 callback 与 ``exec`` 等子命令共用配置和 Runtime builder。它只负责把
    终端选项转换成 TUI 启动参数，不实现 Session 调度或 Agent Loop。
    """

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

        Typer 已选择子命令时立即返回，避免执行 ``codecraft exec`` 等命令时根
        callback 又启动 TUI。``--resume`` 与 ``--last`` 都会改变恢复目标，因此
        二者互斥；具体恢复和 Session 执行仍由 TUI 使用同一个 AgentRuntime 完成。

        Raises:
            typer.BadParameter: 同时提供 ``--resume`` 和 ``--last``。
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
