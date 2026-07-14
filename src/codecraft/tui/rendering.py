from __future__ import annotations

from rich.text import Text

from codecraft.schema.session import SessionConfig


def session_header(config: SessionConfig, width: int) -> Text:
    workspace = config.cwd.name or str(config.cwd)
    candidates = (
        (workspace, f"{config.model_provider}/{config.model}"),
        (workspace, None),
        (None, None),
    )
    for candidate_workspace, model in candidates:
        header = _session_header(candidate_workspace, model)
        if len(header.plain) <= width:
            return header
    return _session_header(None, None)


def runtime_status(
    config: SessionConfig,
    status: str,
    token_usage: dict[str, int],
    width: int,
) -> Text:
    status_style = {
        "idle": "#79c99e",
        "running": "#8da2fb",
        "approval": "#d8b56d",
        "failed": "#ef767a",
        "closed": "#777e87",
        "starting": "#777e87",
    }.get(status, "#a5abb3")
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
        )
        if len(line.plain) <= width:
            return line
    return _runtime_status(status, status_style)


def _session_header(workspace: str | None, model: str | None) -> Text:
    header = Text(no_wrap=True, overflow="ellipsis")
    header.append("CodeCraft", style="bold #f1f3f5")
    if workspace is not None:
        header.append("  ·  ", style="#5f656d")
        header.append(workspace, style="#a5abb3")
    if model is not None:
        header.append("  ")
        header.append(model, style="#777e87")
    return header


def _runtime_status(
    status: str,
    status_style: str,
    *,
    sandbox: str | None = None,
    tokens: str | None = None,
    mcp_count: int = 0,
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
        line.append("  ·  ", style="#4f555d")
        line.append(value, style="#777e87")
    return line
