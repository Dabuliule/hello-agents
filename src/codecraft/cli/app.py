"""CodeCraft 命令行应用入口与可替换的 Runtime 装配 seam。

本模块只创建 Typer 应用、注册命令，并把配置和依赖构造委托给 ``bootstrap``。
CLI/TUI 的测试会 monkeypatch 下列前缀为 ``_`` 的构造函数，因此这里刻意保留
薄包装；真正的 Agent Loop 位于 ``core``，不能在命令注册层复制实现。
"""

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
    """解析一次 CLI 启动所需的完整 Session 配置快照。

    Args:
        source: 创建会话的 CLI 场景，例如一次性 ``exec`` 或交互式 TUI。
        provider: CLI 显式选择的 Provider；``None`` 表示保留较低配置层。
        model: CLI 显式选择的模型；``None`` 表示保留较低配置层。
        codecraft_home: 用户配置、Session 日志和索引的存储根。
        config_path: 可选的最高优先级 TOML 配置文件。
        profile: 可选的用户 profile 名称。
        approval_policy: CLI 审批策略覆盖。
        network: CLI 网络能力覆盖；显式 ``False`` 与未提供 ``None`` 不同。

    Returns:
        已完成全部配置层合并、可持久化到 Session 日志的 ``SessionConfig``。

    这层薄包装是测试 seam：命令测试可以替换配置加载而不接触真实用户目录。
    """
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
    """为一个已解析的 SessionConfig 装配完整 AgentRuntime。

    Args:
        config: 当前会话固定使用的配置快照。

    Returns:
        包含 SessionStore、Provider、Tool、审批和 Observer 的运行时。

    Provider 与 Tool builder 仍从本模块调用，目的是让 CLI 测试能够分别替换
    外部模型和工具集合；资源所有权和关闭逻辑仍由 ``AgentRuntime`` 负责。
    """
    return bootstrap.build_runtime(
        config,
        llm_providers=_build_provider_registry(config),
        tool_registry=_build_tool_registry(config),
    )


def _build_provider_registry(config: SessionConfig) -> LLMProviderRegistry:
    """构造 Provider Registry，并保留供 CLI 测试替换的稳定 seam。"""
    return bootstrap.build_provider_registry(config)


def _provider_api_key_env(config: SessionConfig, provider: str) -> str | None:
    """解析指定 Provider 的 API Key 环境变量名，不读取或返回密钥值。"""
    return bootstrap.provider_api_key_env(config, provider)


def _model_api_key_env(provider: str, configured: str | None) -> str | None:
    """返回显式或 Provider 默认的 API Key 环境变量名。"""
    return bootstrap.model_api_key_env(provider, configured)


def _build_tool_registry(config: SessionConfig | None = None) -> ToolRegistry:
    """构造默认或配置感知的 Tool Registry，并保留 CLI 测试 seam。"""
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
