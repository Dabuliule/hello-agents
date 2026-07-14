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

SESSION_CONFIG_SCHEMA_VERSION = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SessionSource(StrEnum):
    CLI_EXEC = "cli_exec"
    CLI_EVAL = "cli_eval"
    CLI_TUI = "cli_tui"
    TEST = "test"


class EvalSessionContext(BaseModel):
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
    workspace_roots: list[Path]
    codecraft_home: Path

    model: str = Field(min_length=1)
    model_provider: str = Field(min_length=1)
    model_api_key_env: str | None = None
    model_base_url: str | None = None

    approval_policy: ApprovalPolicy
    sandbox_mode: SandboxMode
    network_access: bool = False
    sandbox_backend: SandboxBackendType = SandboxBackendType.AUTO
    sandbox_env_allowlist: list[str] = Field(default_factory=list)
    docker_sandbox: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)
    mcp_servers: dict[str, MCPServerSettings] = Field(default_factory=dict)

    base_instructions: str | None = None
    project_instructions: str | None = None
    user_instructions: str | None = None

    max_tool_calls: int = Field(default=30, ge=1, le=1000)
    max_tool_output_chars: int = Field(default=80_000, ge=1, le=10_000_000)
    turn_timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    tool_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    approval_timeout_seconds: int = Field(default=300, ge=1, le=3600)
    max_context_chars: int = Field(default=400_000, ge=1000, le=20_000_000)
    context_keep_recent_items: int = Field(default=12, ge=1, le=100)
    max_parallel_read_tools: int = Field(default=4, ge=1, le=32)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    evaluation: EvalSessionContext | None = None

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: Path) -> Path:
        resolved = value.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError("cwd must be an existing directory")
        return resolved

    @field_validator("codecraft_home")
    @classmethod
    def normalize_codecraft_home(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("workspace_roots")
    @classmethod
    def validate_workspace_roots(cls, value: list[Path]) -> list[Path]:
        if not value:
            raise ValueError("workspace_roots must include at least one directory")

        roots = [path.expanduser().resolve() for path in value]
        missing = [path for path in roots if not path.exists() or not path.is_dir()]
        if missing:
            raise ValueError(f"workspace_roots must be directories: {missing}")

        return list(dict.fromkeys(roots))

    @field_validator("model_api_key_env")
    @classmethod
    def validate_model_api_key_env(cls, value: str | None) -> str | None:
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
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))

    @field_validator("mcp_servers")
    @classmethod
    def validate_mcp_servers(
        cls, values: dict[str, MCPServerSettings]
    ) -> dict[str, MCPServerSettings]:
        return MCPSettings(servers=values).servers

    @model_validator(mode="after")
    def validate_workspace_boundary(self) -> SessionConfig:
        if not any(
            self.cwd == root or root in self.cwd.parents
            for root in self.workspace_roots
        ):
            raise ValueError("cwd must be inside a workspace root")
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
