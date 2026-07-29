from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from codecraft.schema.tool import ToolEffect


class SandboxMode(StrEnum):
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGER_FULL_ACCESS = "danger_full_access"


class SandboxPolicy(BaseModel):
    """根据 tool effect 判断当前 sandbox 是否允许执行。

    这是比 approval 更底层的限制：如果 sandbox 不允许，用户批准也不会执行。
    """

    model_config = ConfigDict(extra="forbid")

    mode: SandboxMode
    network_access: bool = False

    def evaluate_effects(self, effects: set[ToolEffect]) -> "SandboxEvaluation":
        """检查 tool 声明的 effect 是否被当前 sandbox mode 覆盖。"""
        if ToolEffect.NETWORK in effects and not self.network_access:
            return SandboxEvaluation.deny(
                "network access is disabled by sandbox policy",
                denied_effect=ToolEffect.NETWORK,
            )

        if self.mode == SandboxMode.READ_ONLY:
            denied = effects - {ToolEffect.READ_ONLY}
            if denied:
                return SandboxEvaluation.deny(
                    "read_only sandbox allows read-only tools only",
                    denied_effect=sorted(denied)[0],
                )

        return SandboxEvaluation(
            allowed=True,
            reason="tool effects are allowed by sandbox policy",
        )


class SandboxEvaluation(BaseModel):
    allowed: bool
    reason: str
    denied_effect: ToolEffect | None = None

    @classmethod
    def deny(cls, reason: str, *, denied_effect: ToolEffect) -> "SandboxEvaluation":
        return cls(
            allowed=False,
            reason=reason,
            denied_effect=denied_effect,
        )
