from pathlib import Path

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli import bootstrap
from codecraft.cli.commands import (
    register_demo_command,
    register_eval_command,
    register_exec_command,
    register_index_command,
    register_inspect_command,
    register_mcp_server_command,
    register_retrieval_eval_command,
    register_root_command,
    register_sessions_command,
    register_trace_command,
)
from codecraft.core.runtime import AgentRuntime
from codecraft.llm import LLMProviderRegistry
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import ToolRegistry


app = typer.Typer(
    no_args_is_help=False,
    help="CodeCraft local coding agent.",
)


def _load_session_config(
    *,
    source: SessionSource,
    provider: str | None,
    model: str | None,
    codecraft_home: Path,
    config_path: Path | None,
    profile: str | None,
    approval_policy: ApprovalPolicy | None,
    network: bool | None,
) -> SessionConfig:
    """测试可替换的 CLI 配置加载 seam，委托 bootstrap。"""
    return bootstrap.load_session_config(
        source=source,
        provider=provider,
        model=model,
        codecraft_home=codecraft_home,
        config_path=config_path,
        profile=profile,
        approval_policy=approval_policy,
        network=network,
    )


def _build_runtime(config: SessionConfig) -> AgentRuntime:
    """通过可 monkeypatch 的 Provider/Tool builders 装配 Runtime。"""
    return bootstrap.build_runtime(
        config,
        llm_providers=_build_provider_registry(config),
        tool_registry=_build_tool_registry(config),
    )


def _build_provider_registry(config: SessionConfig) -> LLMProviderRegistry:
    """CLI seam：构造模型 Provider Registry。"""
    return bootstrap.build_provider_registry(config)


def _provider_api_key_env(config: SessionConfig, provider: str) -> str | None:
    """CLI seam：解析指定 Provider 的 API Key 环境名。"""
    return bootstrap.provider_api_key_env(config, provider)


def _model_api_key_env(provider: str, configured: str | None) -> str | None:
    """CLI seam：解析显式或默认 API Key 环境名。"""
    return bootstrap.model_api_key_env(provider, configured)


def _build_tool_registry(config: SessionConfig | None = None) -> ToolRegistry:
    """CLI seam：构造默认或配置感知的 Tool Registry。"""
    return bootstrap.build_tool_registry(config)


register_exec_command(app)
register_demo_command(app)
register_eval_command(app)
register_index_command(app)
register_retrieval_eval_command(app)
register_sessions_command(app)
register_inspect_command(app)
register_mcp_server_command(app)
register_trace_command(app)
register_root_command(app)


if __name__ == "__main__":
    app()
