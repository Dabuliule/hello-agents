"""配置文件合并阶段使用的强类型设置模型。

这些模型描述 TOML 中按 section 组织的用户配置，并在所有配置层合并后执行字段
和跨字段校验。它们是创建 Session 前的中间表示；真正会随事件日志持久化、用于
Resume 的执行快照是 ``schema.session.SessionConfig``。
"""

from __future__ import annotations

from pathlib import Path
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from codecraft.approval.policy import ApprovalPolicy
from codecraft.mcp.config import MCPSettings
from codecraft.sandbox import DockerSandboxConfig, SandboxBackendType, SandboxMode

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ModelSettings(BaseModel):
    """模型连接标识与上下文、输出 Token 预算。

    ``api_key_env`` 只保存环境变量名称，不保存密钥值。上下文窗口和最大输出
    先做各自范围校验，随后由 ``RuntimeSettings`` 检查它们与安全余量的关系。
    """

    provider: str = "qwen"
    name: str = "qwen-plus"
    api_key_env: str | None = None
    base_url: str | None = None
    context_window_tokens: int = Field(default=131_072, ge=4096, le=10_000_000)
    max_output_tokens: int = Field(default=8192, ge=1, le=1_000_000)


class ApprovalSettings(BaseModel):
    """Runtime 默认审批策略。"""

    policy: ApprovalPolicy = ApprovalPolicy.ON_REQUEST


class SandboxSettings(BaseModel):
    """沙箱模式、网络、后端、环境变量和 Docker 配置。"""

    mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    network_access: bool = False
    backend: SandboxBackendType = SandboxBackendType.AUTO
    env_allowlist: list[str] = Field(default_factory=list)
    docker: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)

    @field_validator("env_allowlist")
    @classmethod
    def validate_env_names(cls, values: list[str]) -> list[str]:
        """校验并稳定去重传入沙箱的环境变量名称。"""
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))


class PathsSettings(BaseModel):
    """CodeCraft 用户数据路径设置。"""

    codecraft_home: Path = Path("~/.codecraft")

    @field_validator("codecraft_home")
    @classmethod
    def expand_path(cls, value: Path) -> Path:
        """展开用户主目录符号，保留相对路径是否解析给调用边界决定。"""
        return value.expanduser()


class InstructionSettings(BaseModel):
    """用户级附加模型指令。"""

    user: str | None = None


class TurnSettings(BaseModel):
    """单 Turn 的工具、输出、并发、超时和上下文预算。"""

    max_tool_calls: int = Field(default=30, ge=1, le=1000)
    max_tool_output_chars: int = Field(default=80_000, ge=1, le=10_000_000)
    max_tool_output_tokens: int = Field(default=16_384, ge=32, le=1_000_000)
    turn_timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    tool_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    approval_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    context_safety_margin_tokens: int = Field(default=2048, ge=0, le=1_000_000)
    context_keep_recent_items: int = Field(default=12, ge=1, le=100)
    max_parallel_read_tools: int = Field(default=4, ge=1, le=32)


class RuntimeSettings(BaseModel):
    """配置文件各 section 合并后的强类型 Runtime 设置。

    ``ConfigLoader`` 先以本类默认值为基础递归合并各配置层，最后只调用一次
    ``model_validate``。因此局部 TOML section 可以只覆盖一个叶子字段，同时
    未知字段和跨 section 的非法预算仍会在统一边界被 Pydantic 拒绝。

    Example:
        >>> settings = RuntimeSettings()
        >>> (settings.model.provider, settings.sandbox.network_access)
        ('qwen', False)
    """

    model: ModelSettings = Field(default_factory=ModelSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    instructions: InstructionSettings = Field(default_factory=InstructionSettings)
    turn: TurnSettings = Field(default_factory=TurnSettings)

    @model_validator(mode="after")
    def validate_model_token_budget(self) -> RuntimeSettings:
        """确保模型上下文窗口仍为输入消息保留至少一个 Token 的空间。

        Returns:
            预算关系合法的当前设置对象。

        Raises:
            ValueError: 最大输出与安全余量已经占满或超过上下文窗口。
        """
        reserved = self.model.max_output_tokens + self.turn.context_safety_margin_tokens
        if reserved >= self.model.context_window_tokens:
            raise ValueError(
                "model max_output_tokens plus context safety margin must be smaller "
                "than context_window_tokens"
            )
        return self
