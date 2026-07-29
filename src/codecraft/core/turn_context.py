from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from codecraft.approval.policy import ApprovalPolicy
from codecraft.sandbox.policy import SandboxMode
from codecraft.schema.tool import ToolSpec


class TurnContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    turn_id: str

    cwd: Path

    model: str
    model_provider: str

    approval_policy: ApprovalPolicy
    sandbox_mode: SandboxMode
    network_access: bool
    sandbox_env_allowlist: list[str] = Field(default_factory=list)

    available_tools: list[ToolSpec]

    max_tool_calls: int
    max_tool_output_chars: int
    max_tool_output_tokens: int = Field(default=16_384, ge=32)
    turn_timeout_seconds: int = 1800
    tool_timeout_seconds: int = 300
    approval_timeout_seconds: int = 300
    model_context_window_tokens: int = 131_072
    model_max_output_tokens: int = 8192
    context_safety_margin_tokens: int = 2048
    context_keep_recent_items: int = 12
    max_parallel_read_tools: int = 4

    created_at: datetime
