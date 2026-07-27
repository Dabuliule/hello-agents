from __future__ import annotations

import re
import shlex
from enum import StrEnum

from pydantic import BaseModel


class CommandRisk(StrEnum):
    SAFE = "safe"
    PROMPT = "prompt"
    DENY = "deny"


class CommandDecision(BaseModel):
    risk: CommandRisk
    reason: str
    requires_approval: bool


_SHELL_CHAIN_RE = re.compile(r"(&&|\|\||[;|])")

_BROAD_RM_TARGETS = frozenset({"/", "/*", "~", "*", "..", "../", "/.", "./.."})

_GIT_NETWORK_SUBCOMMANDS = frozenset({"clone", "fetch", "ls-remote", "pull", "push"})


class CommandPolicy:
    """对 shell command 做静态风险分类。

    这里不执行命令，只根据可执行文件、常见子命令、网络需求和明显危险的
    `rm -rf` 目标给出 safe/prompt/deny。真正是否运行还要看 approval 和
    sandbox 的结果。
    """

    SAFE_COMMANDS = {
        "pwd",
        "ls",
        "rg",
        "grep",
        "cat",
        "git",
        "pytest",
        "python",
        "uv",
        "npm",
        "mvn",
    }
    SAFE_READONLY_COMMANDS = {
        "sed",
    }
    PROMPT_COMMANDS = {
        "pip",
        "poetry",
        "curl",
        "wget",
        "ssh",
        "scp",
        "rm",
        "mv",
        "chmod",
        "chown",
        "git-clean",
        "git-commit",
        "git-push",
    }
    DENY_COMMANDS = {
        "sudo",
        "dd",
        "mkfs",
    }
    NETWORK_COMMANDS = {
        "curl",
        "wget",
        "ssh",
        "scp",
    }

    def classify(
        self, command: str, *, network_access: bool = False
    ) -> CommandDecision:
        """返回命令风险等级，以及是否需要用户 approval。"""
        parts = self._split(command)
        if not parts:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="empty command",
                requires_approval=False,
            )

        sub_commands = self._split_raw_command(command)
        if len(sub_commands) > 1:
            # compound command 取子命令中的最高风险，避免安全命令掩盖危险片段。
            return self._classify_compound(sub_commands, network_access=network_access)

        return self._classify_single(parts, network_access=network_access)

    def _classify_single(
        self, parts: list[str], *, network_access: bool
    ) -> CommandDecision:
        executable = parts[0]

        if self._is_destructive_rm(parts):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="destructive rm -rf on broad path is denied",
                requires_approval=False,
            )

        if executable in self.DENY_COMMANDS:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason=f"{executable} is denied",
                requires_approval=False,
            )

        if executable in self.NETWORK_COMMANDS and not network_access:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason=f"{executable} requires network access",
                requires_approval=False,
            )

        if executable == "git" and self._git_requires_network(parts):
            subcommand = parts[1]
            if not network_access:
                return CommandDecision(
                    risk=CommandRisk.DENY,
                    reason=f"git {subcommand} requires network access",
                    requires_approval=False,
                )
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason=f"git {subcommand} requires approval",
                requires_approval=True,
            )

        compound = self._compound_name(parts)
        if compound in self.PROMPT_COMMANDS or executable in self.PROMPT_COMMANDS:
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason=f"{compound} requires approval",
                requires_approval=True,
            )

        if self._is_known_safe(parts):
            return CommandDecision(
                risk=CommandRisk.SAFE,
                reason="command is classified as safe",
                requires_approval=False,
            )

        return CommandDecision(
            risk=CommandRisk.PROMPT,
            reason="unknown command requires approval",
            requires_approval=True,
        )

    def _classify_compound(
        self,
        sub_commands: list[str],
        *,
        network_access: bool,
    ) -> CommandDecision:
        highest: CommandDecision | None = None
        sub_reasons: list[str] = []

        for sub_str in sub_commands:
            sub_str = sub_str.strip()
            if not sub_str:
                continue
            parts = self._split(sub_str)
            if not parts:
                continue
            decision = self._classify_single(parts, network_access=network_access)
            sub_reasons.append(f"{parts[0]}:{decision.risk.value}")
            if highest is None or self._risk_rank(decision.risk) > self._risk_rank(
                highest.risk
            ):
                highest = decision

        if highest is None:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="empty compound command",
                requires_approval=False,
            )

        return CommandDecision(
            risk=highest.risk,
            reason=f"compound command ({'; '.join(sub_reasons)})",
            requires_approval=highest.requires_approval,
        )

    @staticmethod
    def _split(command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return []

    @staticmethod
    def _split_raw_command(command: str) -> list[str]:
        parts = _SHELL_CHAIN_RE.split(command)
        return [
            part
            for part in parts
            if part.strip() and part.strip() not in {"&&", "||", "|", ";"}
        ]

    @staticmethod
    def _compound_name(parts: list[str]) -> str:
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return parts[0]

    @staticmethod
    def _risk_rank(risk: CommandRisk) -> int:
        return {CommandRisk.SAFE: 0, CommandRisk.PROMPT: 1, CommandRisk.DENY: 2}[risk]

    def _is_known_safe(self, parts: list[str]) -> bool:
        """识别项目内常见的只读或测试类命令。"""
        executable = parts[0]

        if executable == "sed":
            return not self._sed_is_inplace(parts)

        if executable not in self.SAFE_COMMANDS:
            return False

        if executable == "git" and len(parts) >= 2:
            return parts[1] in {
                "status",
                "diff",
                "show",
                "log",
                "branch",
                "stash",
                "tag",
                "remote",
            }

        if executable == "uv" and len(parts) >= 3:
            return parts[1:3] in (["run", "pytest"], ["run", "ruff"])

        if executable == "npm" and len(parts) >= 2:
            return parts[1] in {"test", "run"}

        if executable == "python" and len(parts) >= 3:
            return parts[1:3] == ["-m", "pytest"]

        if executable == "python" and len(parts) == 2:
            return parts[1] in {"--version", "-V"}

        return executable in {"pwd", "ls", "rg", "grep", "cat", "pytest", "mvn"}

    @staticmethod
    def _sed_is_inplace(parts: list[str]) -> bool:
        for part in parts[1:]:
            if part in {"-i", "--in-place"}:
                return True
            if part.startswith("-") and not part.startswith("--") and "i" in part:
                return True
        return False

    @staticmethod
    def _is_destructive_rm(parts: list[str]) -> bool:
        """拦截 `rm -rf` 这类针对宽泛路径的破坏性命令。"""
        if not parts or parts[0] != "rm":
            return False

        has_recursive, has_force = CommandPolicy._rm_flags(parts[1:])
        if not (has_recursive and has_force):
            return False

        return CommandPolicy._rm_target(parts[1:]) in _BROAD_RM_TARGETS

    @staticmethod
    def _rm_flags(arguments: list[str]) -> tuple[bool, bool]:
        has_recursive = False
        has_force = False
        for part in arguments:
            if part in {"-r", "-R", "--recursive"}:
                has_recursive = True
                continue
            if part in {"-f", "--force"}:
                has_force = True
                continue
            if part.startswith("-") and not part.startswith("--") and len(part) > 1:
                flags = set(part[1:])
                if "r" in flags or "R" in flags:
                    has_recursive = True
                if "f" in flags:
                    has_force = True
        return has_recursive, has_force

    @staticmethod
    def _rm_target(arguments: list[str]) -> str:
        for part in reversed(arguments):
            if part.startswith("-"):
                continue
            return part
        return ""

    @staticmethod
    def _git_requires_network(parts: list[str]) -> bool:
        if len(parts) < 2:
            return False
        return parts[1] in _GIT_NETWORK_SUBCOMMANDS
