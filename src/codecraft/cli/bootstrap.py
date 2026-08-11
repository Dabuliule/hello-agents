from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.config import ConfigLoader, ConfigOverrides
from codecraft.core.ids import new_id
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.llm import (
    DeepSeekProvider,
    LLMProviderRegistry,
    OpenAIProvider,
    QwenProvider,
)
from codecraft.mcp.client import MCPStdioProvider
from codecraft.prompt import BASE_INSTRUCTIONS
from codecraft.retrieval import (
    ContextEngine,
    LexicalRetriever,
    RepositoryIndex,
    ScanRetriever,
    SymbolRetriever,
    WorkspaceIndexObserver,
)
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.sandbox import SandboxBackendType, build_sandbox_backend
from codecraft.skill import LoadSkillTool, SkillRegistry
from codecraft.tool import (
    ApplyPatchTool,
    BashTool,
    ListFilesTool,
    ReadFileTool,
    ToolRegistry,
    WorkspaceSearchTool,
    WriteFileTool,
)


@dataclass(frozen=True)
class RuntimeBootstrapResult:
    """CLI 配置加载与依赖装配的一次性 SessionConfig/AgentRuntime 对。"""

    config: SessionConfig
    runtime: AgentRuntime


def bootstrap_runtime(
    *,
    source: SessionSource,
    provider: str | None,
    model: str | None,
    codecraft_home: Path,
    config_path: Path | None,
    profile: str | None,
    approval_policy: ApprovalPolicy | None,
    network: bool | None,
) -> RuntimeBootstrapResult:
    """从 CLI 选择加载 SessionConfig 并构造匹配的完整 Runtime。

    Example:
        ``bootstrap_runtime(source=SessionSource.CLI_EXEC, provider=None, ...)``
        会按配置优先级生成新 session_id，并装配 Provider、Tool、MCP 和 Skill。
    """
    config = load_session_config(
        source=source,
        provider=provider,
        model=model,
        codecraft_home=codecraft_home,
        config_path=config_path,
        profile=profile,
        approval_policy=approval_policy,
        network=network,
    )
    return RuntimeBootstrapResult(config=config, runtime=build_runtime(config))


def load_session_config(
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
    """合并默认/用户/profile/项目/显式/CLI 配置为新 Session 快照。

    当前 ``Path.cwd()`` 同时作为配置发现根和 Session cwd；CLI 的 None 表示
    不覆盖较低层，显式 False 仍可关闭网络。Provider 的 API key env 使用
    用户配置优先、已知 Provider 默认名兜底。
    """
    settings = ConfigLoader(
        cwd=Path.cwd(),
        codecraft_home=codecraft_home,
    ).load(
        profile=profile,
        config_path=config_path,
        overrides=ConfigOverrides.from_cli(
            provider=provider,
            model=model,
            approval_policy=approval_policy,
            network_access=network,
            codecraft_home=codecraft_home,
        ),
    )
    cwd = Path.cwd().resolve()
    return SessionConfig(
        session_id=new_id("ses_"),
        source=source,
        cwd=cwd,
        codecraft_home=settings.paths.codecraft_home,
        model=settings.model.name,
        model_provider=settings.model.provider,
        model_api_key_env=model_api_key_env(
            settings.model.provider, settings.model.api_key_env
        ),
        model_base_url=settings.model.base_url,
        model_context_window_tokens=settings.model.context_window_tokens,
        model_max_output_tokens=settings.model.max_output_tokens,
        approval_policy=settings.approval.policy,
        sandbox_mode=settings.sandbox.mode,
        network_access=settings.sandbox.network_access,
        sandbox_backend=settings.sandbox.backend,
        sandbox_env_allowlist=settings.sandbox.env_allowlist,
        docker_sandbox=settings.sandbox.docker,
        mcp_servers=settings.mcp.servers,
        base_instructions=BASE_INSTRUCTIONS,
        user_instructions=settings.instructions.user,
        max_tool_calls=settings.turn.max_tool_calls,
        max_tool_output_chars=settings.turn.max_tool_output_chars,
        max_tool_output_tokens=settings.turn.max_tool_output_tokens,
        turn_timeout_seconds=settings.turn.turn_timeout_seconds,
        tool_timeout_seconds=settings.turn.tool_timeout_seconds,
        approval_timeout_seconds=settings.turn.approval_timeout_seconds,
        context_safety_margin_tokens=settings.turn.context_safety_margin_tokens,
        context_keep_recent_items=settings.turn.context_keep_recent_items,
        max_parallel_read_tools=settings.turn.max_parallel_read_tools,
    )


def build_runtime(
    config: SessionConfig,
    *,
    llm_providers: LLMProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    skill_registry: SkillRegistry | None = None,
) -> AgentRuntime:
    """装配 Store、Provider、Tool、审批、索引 Observer 和同一 SkillRegistry。

    注入自定义 ToolRegistry 时，如果 Skill 非空就确保存在绑定同一个 Registry
    的 LoadSkillTool；名称已被其他 Tool 占用则拒绝，避免 Prompt 与工具激活源
    不一致。
    """
    index = RepositoryIndex(config.codecraft_home / "indexes")
    skills = (
        skill_registry if skill_registry is not None else build_skill_registry(config)
    )
    tools = tool_registry or build_tool_registry(config, skill_registry=skills)
    if tool_registry is not None and skills:
        registered = {tool.name: tool for tool in tools.list()}
        load_skill = registered.get("load_skill")
        if load_skill is None:
            tools.register(LoadSkillTool(skills))
        elif (
            not isinstance(load_skill, LoadSkillTool)
            or load_skill.registry is not skills
        ):
            raise ValueError("load_skill tool must use the runtime skill registry")
    return AgentRuntime(
        session_store=SessionStore(config.codecraft_home),
        llm_providers=llm_providers or build_provider_registry(config),
        tool_registry=tools,
        approval_manager=ApprovalManager(
            reviewer=ThreadApprovalReviewer(),
        ),
        tool_result_observers=[WorkspaceIndexObserver(index)],
        skill_registry=skills,
    )


def build_skill_registry(config: SessionConfig) -> SkillRegistry:
    """从用户级 codecraft_home/skills 与项目 .codecraft/skills 发现 Skill。"""
    return SkillRegistry.discover(
        user_root=config.codecraft_home / "skills",
        project_root=config.cwd / ".codecraft" / "skills",
    )


def build_provider_registry(config: SessionConfig) -> LLMProviderRegistry:
    """注册 OpenAI/Qwen/DeepSeek，并只给当前 Provider 应用配置覆盖。"""
    return LLMProviderRegistry(
        [
            OpenAIProvider(
                api_key_env=provider_api_key_env(config, "openai"),
                base_url=config.model_base_url,
            ),
            QwenProvider(
                api_key_env=provider_api_key_env(config, "qwen"),
                base_url=config.model_base_url,
            ),
            DeepSeekProvider(
                api_key_env=provider_api_key_env(config, "deepseek"),
                base_url=config.model_base_url,
            ),
        ]
    )


def provider_api_key_env(config: SessionConfig, provider: str) -> str | None:
    """非当前 Provider 忽略 Session 的显式 key env，再使用其默认环境名。"""
    configured = config.model_api_key_env if config.model_provider == provider else None
    return model_api_key_env(provider, configured)


def model_api_key_env(provider: str, configured: str | None) -> str | None:
    """返回显式名称，或已知 Provider 的标准 API Key 环境变量。"""
    if configured:
        return configured
    if provider == "qwen":
        return "DASHSCOPE_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    return None


def build_tool_registry(
    config: SessionConfig | None = None,
    *,
    skill_registry: SkillRegistry | None = None,
) -> ToolRegistry:
    """构造内置 Tool、可选索引检索、Sandbox、Skill 与启用的 MCP Provider。

    无 config 的测试/兼容路径使用 ScanRetriever 与显式 ProcessBackend；正常
    Runtime 按配置创建 OS/container 后端，并注册 scan/lexical/symbol。
    """
    if config is None:
        context_engine = ContextEngine()
    else:
        index = RepositoryIndex(config.codecraft_home / "indexes")
        context_engine = ContextEngine(
            [
                ScanRetriever(),
                LexicalRetriever(index),
                SymbolRetriever(index),
            ]
        )
    sandbox_backend = (
        build_sandbox_backend(config.sandbox_backend, config.docker_sandbox)
        if config is not None
        else build_sandbox_backend(SandboxBackendType.PROCESS)
    )
    tools = [
        ReadFileTool(),
        ListFilesTool(),
        WorkspaceSearchTool(context_engine),
        WriteFileTool(),
        ApplyPatchTool(),
        BashTool(sandbox_backend=sandbox_backend),
    ]
    if skill_registry:
        tools.append(LoadSkillTool(skill_registry))
    registry = ToolRegistry(tools)
    if config is not None:
        for server_name, settings in config.mcp_servers.items():
            if settings.enabled:
                registry.register_async_provider(
                    MCPStdioProvider(
                        server_name,
                        settings,
                        workspace_cwd=config.cwd,
                    )
                )
    return registry
