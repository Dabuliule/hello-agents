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
    """对未知远程能力采用 network+external 且需审批的保守默认效果。"""
    return {"network", "external"}


class MCPToolPolicySettings(BaseModel):
    """单个远程工具覆盖的 Tool effects 与审批要求。"""

    effects: set[MCPToolEffect] = Field(default_factory=_default_tool_effects)
    requires_approval: bool = True


class MCPServerSettings(BaseModel):
    """一个 stdio MCP 进程、发现预算和工具治理策略。"""

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
        """当前版本只接受明确实现的 stdio transport。"""
        if value != "stdio":
            raise ValueError("MCP v1 currently supports stdio transport only")
        return value

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: str) -> str:
        """拒绝前导选项或 NUL，确保 command 是单个可执行文件字段。"""
        if value.startswith("-") or "\x00" in value:
            raise ValueError("MCP command must be an executable, not an option")
        return value

    @field_validator("cwd")
    @classmethod
    def expand_cwd(cls, value: Path | None) -> Path | None:
        """配置加载时展开用户目录，绝对/相对解析延迟到 workspace 已知时。"""
        return value.expanduser() if value is not None else None

    @field_validator("env_allowlist")
    @classmethod
    def validate_env_allowlist(cls, values: list[str]) -> list[str]:
        """严格校验环境变量名并按首次出现去重。"""
        invalid = [value for value in values if not _ENV_NAME.fullmatch(value)]
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return list(dict.fromkeys(values))

    def policy_for(self, tool_name: str) -> MCPToolPolicySettings:
        """返回工具级覆盖；缺失时复制服务器默认 effects 和审批策略。"""
        return self.tools.get(
            tool_name,
            MCPToolPolicySettings(
                effects=set(self.default_effects),
                requires_approval=self.requires_approval,
            ),
        )


class MCPSettings(BaseModel):
    """按本地安全名称索引的全部 MCP Server 配置。"""

    servers: dict[str, MCPServerSettings] = Field(default_factory=dict)

    @field_validator("servers")
    @classmethod
    def validate_server_names(
        cls, values: dict[str, MCPServerSettings]
    ) -> dict[str, MCPServerSettings]:
        """限制名称为 1-24 位安全字符，保证本地 Tool 名可预测。"""
        invalid = [name for name in values if not _SERVER_NAME.fullmatch(name)]
        if invalid:
            raise ValueError(
                f"MCP server names must use 1-24 letters, digits, '_' or '-': {invalid}"
            )
        return values
