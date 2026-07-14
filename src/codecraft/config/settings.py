from __future__ import annotations

from pathlib import Path
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from codecraft.approval.policy import ApprovalPolicy
from codecraft.mcp.config import MCPSettings
from codecraft.sandbox import DockerSandboxConfig, SandboxBackendType, SandboxMode

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ModelSettings(BaseModel):
    provider: str = "qwen"
    name: str = "qwen-plus"
    api_key_env: str | None = None
    base_url: str | None = None
    context_window_tokens: int = Field(default=131_072, ge=4096, le=10_000_000)
    max_output_tokens: int = Field(default=8192, ge=1, le=1_000_000)


class ApprovalSettings(BaseModel):
    policy: ApprovalPolicy = ApprovalPolicy.ON_REQUEST


class SandboxSettings(BaseModel):
    mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    network_access: bool = False
    backend: SandboxBackendType = SandboxBackendType.AUTO
    env_allowlist: list[str] = Field(default_factory=list)
    docker: DockerSandboxConfig = Field(default_factory=DockerSandboxConfig)

    @field_validator("env_allowlist")
    @classmethod
    def validate_env_names(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))


class PathsSettings(BaseModel):
    codecraft_home: Path = Path("~/.codecraft")

    @field_validator("codecraft_home")
    @classmethod
    def expand_path(cls, value: Path) -> Path:
        return value.expanduser()


class InstructionSettings(BaseModel):
    user: str | None = None


class TurnSettings(BaseModel):
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
    model: ModelSettings = Field(default_factory=ModelSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    instructions: InstructionSettings = Field(default_factory=InstructionSettings)
    turn: TurnSettings = Field(default_factory=TurnSettings)

    @model_validator(mode="after")
    def validate_model_token_budget(self) -> RuntimeSettings:
        reserved = self.model.max_output_tokens + self.turn.context_safety_margin_tokens
        if reserved >= self.model.context_window_tokens:
            raise ValueError(
                "model max_output_tokens plus context safety margin must be smaller "
                "than context_window_tokens"
            )
        return self
