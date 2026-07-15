from __future__ import annotations

from rich.text import Text

from codecraft.schema.session import SessionConfig
from codecraft.tui.theme import TUIColorPalette


def session_header(
    config: SessionConfig,
    width: int,
    palette: TUIColorPalette,
) -> Text:
    workspace = config.cwd.name or str(config.cwd)
    candidates = (
        (workspace, f"{config.model_provider}/{config.model}"),
        (workspace, None),
        (None, None),
    )
    for candidate_workspace, model in candidates:
        header = _session_header(candidate_workspace, model, palette)
        if len(header.plain) <= width:
            return header
    return _session_header(None, None, palette)


def runtime_status(
    config: SessionConfig,
    status: str,
    token_usage: dict[str, int],
    width: int,
    palette: TUIColorPalette,
) -> Text:
    status_style = {
        "idle": palette.success,
        "running": palette.accent,
        "approval": palette.warning,
        "failed": palette.error,
        "closed": palette.faint,
        "starting": palette.faint,
    }.get(status, palette.secondary)
    candidates = (
        (True, True),
        (False, True),
        (False, False),
    )
    for include_sandbox, include_tokens in candidates:
        line = _runtime_status(
            status,
            status_style,
            sandbox=str(config.sandbox_mode) if include_sandbox else None,
            tokens=(
                f"{token_usage['total_tokens']:,} tokens" if include_tokens else None
            ),
            mcp_count=len(config.mcp_servers) if include_sandbox else 0,
            palette=palette,
        )
        if len(line.plain) <= width:
            return line
    return _runtime_status(status, status_style, palette=palette)


def _session_header(
    workspace: str | None,
    model: str | None,
    palette: TUIColorPalette,
) -> Text:
    header = Text(no_wrap=True, overflow="ellipsis")
    header.append("CodeCraft", style=f"bold {palette.strong}")
    if workspace is not None:
        header.append("  ·  ", style=palette.separator)
        header.append(workspace, style=palette.secondary)
    if model is not None:
        header.append("  ")
        header.append(model, style=palette.faint)
    return header


def _runtime_status(
    status: str,
    status_style: str,
    *,
    sandbox: str | None = None,
    tokens: str | None = None,
    mcp_count: int = 0,
    palette: TUIColorPalette,
) -> Text:
    line = Text(no_wrap=True, overflow="ellipsis")
    line.append(status, style=f"bold {status_style}")
    for value in (
        sandbox,
        tokens,
        f"{mcp_count} MCP" if mcp_count else None,
    ):
        if value is None:
            continue
        line.append("  ·  ", style=palette.separator)
        line.append(value, style=palette.faint)
    return line
