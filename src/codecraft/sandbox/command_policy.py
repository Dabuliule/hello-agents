from __future__ import annotations

from dataclasses import dataclass
import posixpath
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


@dataclass(frozen=True)
class _ShellScan:
    segments: tuple[str, ...]
    operators: tuple[str, ...]
    has_expansion: bool
    has_substitution: bool


_BROAD_RM_TARGETS = frozenset(
    {"/", "//", "/*", ".", "./", "./*", "~", "*", "..", "../", "../*"}
)

_GIT_NETWORK_SUBCOMMANDS = frozenset({"clone", "fetch", "ls-remote", "pull", "push"})

_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}
)

_NETWORK_COMMANDS = frozenset({"curl", "wget", "ssh", "scp"})

_DENIED_COMMANDS = frozenset({"sudo", "dd", "mkfs"})

_EXACT_READ_ONLY_COMMANDS = frozenset(
    {
        ("pwd",),
        ("pwd", "-L"),
        ("pwd", "-P"),
        ("python", "--version"),
        ("python", "-V"),
        ("python3", "--version"),
        ("python3", "-V"),
        ("git", "status"),
        ("git", "branch"),
        ("git", "branch", "--show-current"),
        ("git", "stash", "list"),
        ("git", "tag"),
        ("git", "tag", "--list"),
        ("git", "remote"),
        ("git", "remote", "-v"),
    }
)

_READ_ONLY_LS_FLAGS = frozenset("aAlh1Fp")

_READ_ONLY_RG_FLAGS = frozenset(
    {
        "-F",
        "--fixed-strings",
        "-i",
        "--ignore-case",
        "-n",
        "--line-number",
        "-w",
        "--word-regexp",
        "-x",
        "--line-regexp",
    }
)

_DOUBLE_SHELL_OPERATORS = frozenset(
    {"&&", "||", ">>", "<<", "<&", ">&", "|&", ";;", ";&"}
)


class CommandPolicy:
    """Classify shell commands with a small, fail-closed read-only allowlist.

    A command is safe only when its complete argv matches a rule below. Unknown
    arguments and shell evaluation features require approval; opaque command or
    process substitution is denied. The sandbox remains the execution boundary.
    """

    def classify(
        self, command: str, *, network_access: bool = False
    ) -> CommandDecision:
        """Return the risk and approval requirement for one shell command."""
        scan = _scan_shell(command)
        if scan.has_substitution:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="shell command or process substitution is denied",
                requires_approval=False,
            )

        decisions: list[CommandDecision] = []
        parsed_segments: list[list[str]] = []
        for segment in scan.segments:
            parts = self._split(segment)
            if not parts:
                return CommandDecision(
                    risk=CommandRisk.DENY,
                    reason="invalid shell syntax",
                    requires_approval=False,
                )
            parsed_segments.append(parts)
            decisions.append(
                self._classify_single(parts, network_access=network_access)
            )

        if not decisions:
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="empty command",
                requires_approval=False,
            )

        denied = next(
            (decision for decision in decisions if decision.risk == CommandRisk.DENY),
            None,
        )
        if denied is not None:
            return denied

        if scan.has_expansion and any(
            self._is_recursive_force_rm(parts) for parts in parsed_segments
        ):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="rm -rf with shell expansion is denied",
                requires_approval=False,
            )

        if scan.operators:
            operators = ", ".join(scan.operators)
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason=f"shell control syntax requires approval: {operators}",
                requires_approval=True,
            )

        if scan.has_expansion:
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason="shell expansion requires approval",
                requires_approval=True,
            )

        return decisions[0]

    def _classify_single(
        self, parts: list[str], *, network_access: bool
    ) -> CommandDecision:
        executable = self._command_name(parts[0])

        if self._is_destructive_rm(parts):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="destructive rm -rf on broad path is denied",
                requires_approval=False,
            )

        if executable in _DENIED_COMMANDS or executable.startswith("mkfs."):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason=f"{executable} is denied",
                requires_approval=False,
            )

        if executable in _NETWORK_COMMANDS:
            if not network_access:
                return CommandDecision(
                    risk=CommandRisk.DENY,
                    reason=f"{executable} requires network access",
                    requires_approval=False,
                )
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason=f"{executable} requires approval",
                requires_approval=True,
            )

        git_subcommand = self._git_subcommand(parts) if executable == "git" else None
        if git_subcommand in _GIT_NETWORK_SUBCOMMANDS:
            if not network_access:
                return CommandDecision(
                    risk=CommandRisk.DENY,
                    reason=f"git {git_subcommand} requires network access",
                    requires_approval=False,
                )
            return CommandDecision(
                risk=CommandRisk.PROMPT,
                reason=f"git {git_subcommand} requires approval",
                requires_approval=True,
            )

        if self._is_exact_read_only(parts):
            return CommandDecision(
                risk=CommandRisk.SAFE,
                reason="command matches the read-only allowlist",
                requires_approval=False,
            )

        return CommandDecision(
            risk=CommandRisk.PROMPT,
            reason="command is not on the exact read-only allowlist",
            requires_approval=True,
        )

    @staticmethod
    def _split(command: str) -> list[str]:
        try:
            return shlex.split(command)
        except ValueError:
            return []

    @staticmethod
    def _is_exact_read_only(parts: list[str]) -> bool:
        command = tuple(parts)
        if command in _EXACT_READ_ONLY_COMMANDS:
            return True

        if parts[0] == "ls":
            return CommandPolicy._is_read_only_ls(parts[1:])

        if parts[0] == "rg":
            return CommandPolicy._is_read_only_rg(parts[1:])

        return False

    @staticmethod
    def _is_read_only_ls(arguments: list[str]) -> bool:
        if not arguments:
            return True
        return all(
            argument.startswith("-")
            and not argument.startswith("--")
            and len(argument) > 1
            and set(argument[1:]) <= _READ_ONLY_LS_FLAGS
            for argument in arguments
        )

    @staticmethod
    def _is_read_only_rg(arguments: list[str]) -> bool:
        if arguments == ["--files"]:
            return True
        if len(arguments) == 1:
            return bool(arguments[0]) and not arguments[0].startswith("-")
        if len(arguments) == 2 and arguments[0] in _READ_ONLY_RG_FLAGS:
            return bool(arguments[1]) and not arguments[1].startswith("-")
        return False

    @staticmethod
    def _is_destructive_rm(parts: list[str]) -> bool:
        if not CommandPolicy._is_recursive_force_rm(parts):
            return False

        return any(
            posixpath.normpath(target) in _BROAD_RM_TARGETS
            for target in CommandPolicy._rm_targets(parts[1:])
        )

    @staticmethod
    def _is_recursive_force_rm(parts: list[str]) -> bool:
        if not parts or CommandPolicy._command_name(parts[0]) != "rm":
            return False
        has_recursive, has_force = CommandPolicy._rm_flags(parts[1:])
        return has_recursive and has_force

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
    def _rm_targets(arguments: list[str]) -> list[str]:
        targets: list[str] = []
        parsing_options = True
        for part in arguments:
            if parsing_options and part == "--":
                parsing_options = False
                continue
            if parsing_options and part.startswith("-"):
                continue
            targets.append(part)
        return targets

    @staticmethod
    def _git_subcommand(parts: list[str]) -> str | None:
        index = 1
        while index < len(parts):
            part = parts[index]
            if part == "--":
                index += 1
                return parts[index] if index < len(parts) else None
            if part in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
                index += 2
                continue
            if part.startswith(("-C", "-c")) and len(part) > 2:
                index += 1
                continue
            if part.startswith(
                (
                    "--git-dir=",
                    "--work-tree=",
                    "--namespace=",
                    "--config-env=",
                )
            ):
                index += 1
                continue
            if part.startswith("-"):
                index += 1
                continue
            return part
        return None

    @staticmethod
    def _command_name(executable: str) -> str:
        return executable.replace("\\", "/").rsplit("/", 1)[-1]


class _ShellScanner:
    def __init__(self, command: str) -> None:
        self.command = command
        self.segments: list[str] = []
        self.operators: list[str] = []
        self.current: list[str] = []
        self.quote: str | None = None
        self.escaped = False
        self.has_expansion = False
        self.has_substitution = False
        self.index = 0

    def scan(self) -> _ShellScan:
        while self.index < len(self.command):
            character = self.command[self.index]
            following = (
                self.command[self.index + 1]
                if self.index + 1 < len(self.command)
                else ""
            )
            if self.escaped:
                self._consume_escaped(character)
            elif self.quote is not None:
                self._consume_quoted(character, following)
            else:
                self._consume_unquoted(character, following)

        self._flush_segment()
        return _ShellScan(
            segments=tuple(self.segments),
            operators=tuple(self.operators),
            has_expansion=self.has_expansion,
            has_substitution=self.has_substitution,
        )

    def _consume_escaped(self, character: str) -> None:
        self.current.append(character)
        self.escaped = False
        self.index += 1

    def _consume_quoted(self, character: str, following: str) -> None:
        self.current.append(character)
        if character == self.quote:
            self.quote = None
        elif self.quote == '"' and character == "`":
            self.has_substitution = True
        elif self.quote == '"' and character == "$":
            if following == "(":
                self.has_substitution = True
            else:
                self.has_expansion = True
        self.index += 1

    def _consume_unquoted(self, character: str, following: str) -> None:
        if character == "\\":
            self.current.append(character)
            self.escaped = True
            self.index += 1
            return
        if character in {"'", '"'}:
            self.quote = character
            self.current.append(character)
            self.index += 1
            return
        if character == "`":
            self.has_substitution = True
            self.current.append(character)
            self.index += 1
            return
        if character == "$":
            self._consume_dollar(following)
            return
        if character in {"<", ">"} and following == "(":
            self.has_substitution = True
            self.current.append(character)
            self.index += 1
            return
        if character in {"*", "?", "[", "{", "~"}:
            self.has_expansion = True
            self.current.append(character)
            self.index += 1
            return
        if character in {"\n", "\r"}:
            self._consume_newline(character, following)
            return
        if character in {";", "&", "|", "<", ">", "(", ")"}:
            self._consume_operator(character, following)
            return
        self.current.append(character)
        self.index += 1

    def _consume_dollar(self, following: str) -> None:
        if following == "(":
            self.has_substitution = True
        else:
            self.has_expansion = True
        self.current.append("$")
        self.index += 1

    def _consume_newline(self, character: str, following: str) -> None:
        self._flush_segment()
        self._add_operator("newline")
        if character == "\r" and following == "\n":
            self.index += 1
        self.index += 1

    def _consume_operator(self, character: str, following: str) -> None:
        self._flush_segment()
        pair = f"{character}{following}"
        if pair in _DOUBLE_SHELL_OPERATORS:
            self._add_operator(pair)
            self.index += 2
        else:
            self._add_operator(character)
            self.index += 1

    def _flush_segment(self) -> None:
        segment = "".join(self.current).strip()
        if segment:
            self.segments.append(segment)
        self.current.clear()

    def _add_operator(self, operator: str) -> None:
        if operator not in self.operators:
            self.operators.append(operator)


def _scan_shell(command: str) -> _ShellScan:
    return _ShellScanner(command).scan()
