from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
MCPToolEffect = Literal[
    "read_only",
    "workspace_write",
    "process_exec",
    "network",
    "external",
]


def _default_tool_effects() -> set[MCPToolEffect]:
    return {"network", "external"}


class MCPToolPolicySettings(BaseModel):
    effects: set[MCPToolEffect] = Field(default_factory=_default_tool_effects)
    requires_approval: bool = True


class MCPServerSettings(BaseModel):
    enabled: bool = True
    transport: str = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: Path | None = None
    env_allowlist: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_tools: int = Field(default=128, ge=1, le=1024)
    max_pages: int = Field(default=32, ge=1, le=1024)
    max_discovery_bytes: int = Field(default=1_000_000, ge=1024, le=100_000_000)
    default_effects: set[MCPToolEffect] = Field(default_factory=_default_tool_effects)
    requires_approval: bool = True
    tools: dict[str, MCPToolPolicySettings] = Field(default_factory=dict)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, value: str) -> str:
        if value != "stdio":
            raise ValueError("MCP v1 currently supports stdio transport only")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        if value.startswith("-") or "\x00" in value:
            raise ValueError("MCP command must be an executable, not an option")
        return value

    @field_validator("cwd")
    @classmethod
    def expand_cwd(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @field_validator("env_allowlist")
    @classmethod
    def validate_env_allowlist(cls, values: list[str]) -> list[str]:
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))

    def policy_for(self, tool_name: str) -> MCPToolPolicySettings:
        return self.tools.get(
            tool_name,
            MCPToolPolicySettings(
                effects=set(self.default_effects),
                requires_approval=self.requires_approval,
            ),
        )


class MCPSettings(BaseModel):
    servers: dict[str, MCPServerSettings] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def validate_server_names(
        cls, values: dict[str, MCPServerSettings]
    ) -> dict[str, MCPServerSettings]:
        invalid = [name for name in values if not _SERVER_NAME.fullmatch(name)]
        if invalid:
            raise ValueError(
                f"MCP server names must use 1-24 letters, digits, '_' or '-': {invalid}"
            )
        return values
