from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codecraft.approval.policy import ApprovalPolicy
from codecraft.mcp.config import MCPServerSettings, MCPSettings
from codecraft.sandbox import DockerSandboxConfig, SandboxBackendType, SandboxMode
from codecraft.schema.event import RuntimeEvent

SESSION_CONFIG_SCHEMA_VERSION: Literal[1] = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SessionSource(StrEnum):
    """创建 Session 的 CLI、评测、TUI 或测试入口。"""

    CLI_EXEC = "cli_exec"
    CLI_DEMO = "cli_demo"
    CLI_EVAL = "cli_eval"
    CLI_TUI = "cli_tui"
    TEST = "test"


class EvalSessionContext(BaseModel):
    """评测 Session 与 run、task、attempt 的关联信息。"""

    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt: int = Field(ge=1)


class SessionConfig(BaseModel):
    """启动或恢复 session 所需的完整运行配置。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SESSION_CONFIG_SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    source: SessionSource

    cwd: Path
    codecraft_home: Path

    model: str = Field(min_length=1)
    model_provider: str = Field(min_length=1)
    model_api_key_env: str | None = None
    model_base_url: str | None = None
    model_context_window_tokens: int = Field(
        default=131_072,
        ge=4096,
        le=10_000_000,
    )
    model_max_output_tokens: int = Field(default=8192, ge=1, le=1_000_000)

    approval_policy: ApprovalPolicy
    sandbox_mode: SandboxMode
    network_access: bool = False
    sandbox_backend: SandboxBackendType = SandboxBackendType.AUTO
    sandbox_env_allowlist: list[str] = Field(default_factory=list)
    docker_sandbox: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)
    mcp_servers: dict[str, MCPServerSettings] = Field(default_factory=dict)

    base_instructions: str | None = None
    user_instructions: str | None = None

    max_tool_calls: int = Field(default=30, ge=1, le=1000)
    max_tool_output_chars: int = Field(default=80_000, ge=1, le=10_000_000)
    max_tool_output_tokens: int = Field(default=16_384, ge=32, le=1_000_000)
    turn_timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    tool_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    approval_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    context_safety_margin_tokens: int = Field(default=2048, ge=0, le=1_000_000)
    context_keep_recent_items: int = Field(default=12, ge=1, le=100)
    max_parallel_read_tools: int = Field(default=4, ge=1, le=32)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evaluation: EvalSessionContext | None = None

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: Path) -> Path:
        """展开并解析工作目录，但把存在性检查延迟到执行边界。"""
        return value.expanduser().resolve()

    def ensure_runtime_ready(self) -> None:
        """Validate environment-dependent preconditions at an execution boundary."""
        if not self.cwd.exists() or not self.cwd.is_dir():
            raise ValueError("cwd must be an existing directory")

    @field_validator("codecraft_home")
    @classmethod
    def normalize_codecraft_home(cls, value: Path) -> Path:
        """把 CodeCraft 数据目录规范为展开后的绝对路径。"""
        return value.expanduser().resolve()

    @field_validator("model_api_key_env")
    @classmethod
    def validate_model_api_key_env(cls, value: str | None) -> str | None:
        """仅允许合法环境变量标识符，避免把密钥值误填进配置。"""
        if value is not None and not _ENV_NAME.fullmatch(value):
            raise ValueError("model_api_key_env must be an environment variable name")
        return value

    @field_validator("model_provider")
    @classmethod
    def normalize_model_provider(cls, value: str) -> str:
        """Provider 名称属于程序标识符，统一使用无空白的小写形式。"""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("model_provider must not be empty")
        return normalized

    @field_validator("sandbox_env_allowlist")
    @classmethod
    def validate_sandbox_env_names(cls, values: list[str]) -> list[str]:
        """校验并按首次出现顺序去重沙箱环境变量 allowlist。"""
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))

    @field_validator("mcp_servers")
    @classmethod
    def validate_mcp_servers(
        cls, values: dict[str, MCPServerSettings]
    ) -> dict[str, MCPServerSettings]:
        """复用 ``MCPSettings`` 校验 server 名称与配置。"""
        return MCPSettings(servers=values).servers

    @model_validator(mode="after")
    def validate_model_token_budget(self) -> SessionConfig:
        """确保固定输出预算和安全余量没有吃完模型上下文窗口。"""
        reserved = self.model_max_output_tokens + self.context_safety_margin_tokens
        if reserved >= self.model_context_window_tokens:
            raise ValueError(
                "model output tokens plus context safety margin must be smaller "
                "than the model context window"
            )
        return self


class SessionSummary(BaseModel):
    """用于列表页/命令输出的轻量 session 信息。"""

    session_id: str
    path: Path
    valid: bool = True
    error_code: str | None = None
    error_message: str | None = None
    cwd: Path | None = None
    source: SessionSource | None = None
    created_at: datetime | None = None
    last_event_at: datetime | None = None
    event_count: int = 0


class SessionSnapshot(BaseModel):
    """恢复 session 时读取到的配置和事件日志。"""

    config: SessionConfig
    events: list[RuntimeEvent]
