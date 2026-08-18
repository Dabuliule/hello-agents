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

    Args:
        source: 当前会话来自一次性 CLI、TUI 或其他受支持入口。
        provider: 可选的 Provider CLI 覆盖。
        model: 可选的模型 CLI 覆盖。
        codecraft_home: 用户配置、Session 日志、Skill 和索引的存储根。
        config_path: 可选的最高优先级 TOML 配置。
        profile: 可选的用户 profile。
        approval_policy: 可选的审批策略覆盖。
        network: 可选的网络能力覆盖。

    Returns:
        使用同一份解析结果构造的 ``SessionConfig`` 与 ``AgentRuntime`` 对。

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

    Returns:
        包含新 Session ID、权限边界和全部执行预算的持久化配置快照。

    配置在创建 Session 前解析为快照，后续 Turn 不重新读取磁盘配置。这保证
    同一 Session 的执行语义稳定，也让 Resume 能恢复当时真正使用的规则。
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

    Args:
        config: 已解析且将在 Session 中持久化的配置快照。
        llm_providers: 可选的自定义 Provider Registry，主要用于测试和嵌入场景。
        tool_registry: 可选的自定义 Tool Registry。
        skill_registry: 可选的预构造 Skill Registry。

    Returns:
        可以创建或恢复 AgentThread 的完整运行时。

    Raises:
        ValueError: 自定义 Tool Registry 中的 ``load_skill`` 与当前 Skill Registry
            不一致。该检查防止 Prompt 展示的 Skill 与工具实际加载源发生漂移。
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
    """发现用户级与项目级 Skill，并合并为当前 Runtime 的唯一 Registry。

    Args:
        config: 提供 ``codecraft_home`` 和 Session workspace 根的配置快照。

    Returns:
        已扫描 ``codecraft_home/skills`` 与 ``cwd/.codecraft/skills`` 的 Registry。

    Prompt 目录和 ``load_skill`` 工具必须共享这一对象，否则模型看到的可用 Skill
    可能与工具实际可加载的 Skill 不一致。
    """
    return SkillRegistry.discover(
        user_root=config.codecraft_home / "skills",
        project_root=config.cwd / ".codecraft" / "skills",
    )


def build_provider_registry(config: SessionConfig) -> LLMProviderRegistry:
    """注册全部内置 Provider，并为当前 Session 解析连接配置。

    Args:
        config: 指定当前 Provider、API Key 环境变量名和可选 base URL 的快照。

    Returns:
        名称唯一的 OpenAI、Qwen、DeepSeek Provider Registry。

    Runtime 创建 Thread 时按 ``model_provider`` 取一个 Provider。显式 API Key
    环境变量名只属于当前 Provider；其他 Provider 保留各自默认名称，避免一次
    Session 的配置误传给不同服务。这里只传递环境变量名，不读取或持久化密钥。
    """
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
    """为 Registry 中一个 Provider 选择显式或默认 API Key 环境变量名。

    当前 Session 选中的 Provider 可以使用 ``model_api_key_env``；其他 Provider
    必须忽略它并使用自己的默认名称，防止把例如 Qwen 密钥名称传给 OpenAI。
    """
    configured = config.model_api_key_env if config.model_provider == provider else None
    return model_api_key_env(provider, configured)


def model_api_key_env(provider: str, configured: str | None) -> str | None:
    """返回显式名称，或已知 Provider 的标准 API Key 环境变量名。

    未知 Provider 且没有显式配置时返回 ``None``，让具体 Provider 在自身配置
    边界给出可操作错误；本函数始终不读取环境变量的值。
    """
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

    Args:
        config: 可选 Session 快照；缺失时启用不依赖用户目录的兼容工具集。
        skill_registry: 可选的共享 Skill Registry。

    Returns:
        包含内置工具、可选 ``load_skill`` 和启用 MCP Provider 的 Tool Registry。

    有配置的生产路径根据 Session 选择真正的沙箱后端，并让 workspace search
    同时具备实时 scan、SQLite lexical 和 symbol 检索。MCP 先注册为异步
    Provider，工具发现和连接生命周期由 ``ToolRegistry.start/close`` 管理。
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
