from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import posixpath
import re
import shlex

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

_COMMAND_WRAPPERS = frozenset({"command", "exec", "env"})

_SHELL_WRAPPERS = frozenset({"bash", "dash", "ksh", "sh", "zsh"})

_MAX_WRAPPER_DEPTH = 8

_ENV_OPTIONS_WITH_VALUE = frozenset({"-C", "-u", "-a", "--chdir", "--unset"})

_ENV_OPTIONS_WITHOUT_VALUE = frozenset(
    {"-", "-0", "-i", "-v", "--debug", "--ignore-environment", "--null"}
)

_SHELL_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")

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
        return self._classify_command(
            command,
            network_access=network_access,
            wrapper_depth=0,
        )

    def _classify_command(
        self,
        command: str,
        *,
        network_access: bool,
        wrapper_depth: int,
    ) -> CommandDecision:
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
                self._classify_single(
                    parts,
                    network_access=network_access,
                    wrapper_depth=wrapper_depth,
                    has_expansion=scan.has_expansion,
                )
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
        self,
        parts: list[str],
        *,
        network_access: bool,
        wrapper_depth: int,
        has_expansion: bool,
    ) -> CommandDecision:
        wrapper_decision = self._classify_indirection(
            parts,
            network_access=network_access,
            wrapper_depth=wrapper_depth,
            has_expansion=has_expansion,
        )
        if wrapper_decision is not None:
            return wrapper_decision

        executable = self._command_name(parts[0])

        if self._is_destructive_rm(parts):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="destructive rm -rf on broad path is denied",
                requires_approval=False,
            )

        if has_expansion and self._is_recursive_force_rm(parts):
            return CommandDecision(
                risk=CommandRisk.DENY,
                reason="rm -rf with shell expansion is denied",
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

    def _classify_indirection(
        self,
        parts: list[str],
        *,
        network_access: bool,
        wrapper_depth: int,
        has_expansion: bool,
    ) -> CommandDecision | None:
        assignment_count = self._leading_assignment_count(parts)
        if assignment_count:
            inner_parts = parts[assignment_count:]
            if not inner_parts:
                return self._opaque_wrapper_decision(
                    "environment assignment has no command"
                )
            return self._classify_wrapped_parts(
                inner_parts,
                wrapper="environment assignment",
                network_access=network_access,
                wrapper_depth=wrapper_depth,
                has_expansion=has_expansion,
            )

        executable = self._command_name(parts[0])
        if executable in _COMMAND_WRAPPERS:
            unwrapped_parts = self._unwrap_command(executable, parts)
            if unwrapped_parts is None:
                return self._opaque_wrapper_decision(
                    f"unsupported or incomplete {executable} wrapper"
                )
            return self._classify_wrapped_parts(
                unwrapped_parts,
                wrapper=executable,
                network_access=network_access,
                wrapper_depth=wrapper_depth,
                has_expansion=has_expansion,
            )

        if executable in _SHELL_WRAPPERS:
            script = self._unwrap_shell_script(parts)
            if script is None:
                return self._opaque_wrapper_decision(
                    f"unsupported or incomplete {executable} wrapper"
                )
            if has_expansion and self._contains_expansion_marker(script):
                return self._opaque_wrapper_decision(
                    f"{executable} script is controlled by shell expansion"
                )
            if wrapper_depth >= _MAX_WRAPPER_DEPTH:
                return self._wrapper_depth_decision()
            decision = self._classify_command(
                script,
                network_access=network_access,
                wrapper_depth=wrapper_depth + 1,
            )
            return self._elevate_wrapper_decision(decision, executable)
        if has_expansion and self._contains_expansion_marker(parts[0]):
            return self._opaque_wrapper_decision(
                "command name is controlled by shell expansion"
            )
        return None

    def _classify_wrapped_parts(
        self,
        parts: list[str],
        *,
        wrapper: str,
        network_access: bool,
        wrapper_depth: int,
        has_expansion: bool,
    ) -> CommandDecision:
        if wrapper_depth >= _MAX_WRAPPER_DEPTH:
            return self._wrapper_depth_decision()
        if has_expansion and self._contains_expansion_marker(parts[0]):
            return self._opaque_wrapper_decision(
                f"{wrapper} command is controlled by shell expansion"
            )
        decision = self._classify_single(
            parts,
            network_access=network_access,
            wrapper_depth=wrapper_depth + 1,
            has_expansion=has_expansion,
        )
        return self._elevate_wrapper_decision(decision, wrapper)

    @staticmethod
    def _elevate_wrapper_decision(
        decision: CommandDecision, wrapper: str
    ) -> CommandDecision:
        if decision.risk == CommandRisk.DENY:
            return decision
        return CommandDecision(
            risk=CommandRisk.PROMPT,
            reason=f"{wrapper} wrapper requires approval",
            requires_approval=True,
        )

    @staticmethod
    def _opaque_wrapper_decision(reason: str) -> CommandDecision:
        return CommandDecision(
            risk=CommandRisk.DENY,
            reason=reason,
            requires_approval=False,
        )

    @staticmethod
    def _wrapper_depth_decision() -> CommandDecision:
        return CommandDecision(
            risk=CommandRisk.DENY,
            reason=f"command wrapper nesting exceeds {_MAX_WRAPPER_DEPTH} levels",
            requires_approval=False,
        )

    @staticmethod
    def _leading_assignment_count(parts: list[str]) -> int:
        index = 0
        while index < len(parts) and _SHELL_ASSIGNMENT.match(parts[index]):
            index += 1
        return index

    @classmethod
    def _unwrap_command(cls, executable: str, parts: list[str]) -> list[str] | None:
        if executable == "env":
            return cls._unwrap_env(parts)
        if executable == "command":
            return cls._unwrap_command_builtin(parts)
        return cls._unwrap_exec(parts)

    @classmethod
    def _unwrap_env(cls, parts: list[str]) -> list[str] | None:
        index = 1
        parsing_options = True
        while index < len(parts):
            part = parts[index]
            if cls._is_assignment(part):
                index += 1
                continue
            if parsing_options and part == "--":
                parsing_options = False
                index += 1
                continue
            if not parsing_options or not part.startswith("-"):
                break
            option_end = cls._env_option_end(parts, index)
            if option_end is None:
                return None
            index = option_end
        return parts[index:] or None

    @staticmethod
    def _env_option_end(parts: list[str], index: int) -> int | None:
        part = parts[index]
        if part in {"-S", "--split-string"} or part.startswith(
            ("-S", "--split-string=")
        ):
            return None
        if part in _ENV_OPTIONS_WITHOUT_VALUE:
            return index + 1
        if part in _ENV_OPTIONS_WITH_VALUE:
            return index + 2 if index + 1 < len(parts) else None
        if part.startswith(("--chdir=", "--unset=", "--argv0=")):
            return index + 1
        if part.startswith(("-C", "-u")) and len(part) > 2:
            return index + 1
        return None

    @staticmethod
    def _unwrap_command_builtin(parts: list[str]) -> list[str] | None:
        index = 1
        while index < len(parts):
            part = parts[index]
            if part == "--":
                index += 1
                break
            if part == "-p":
                index += 1
                continue
            if part.startswith("-"):
                return None
            break
        return parts[index:] or None

    @staticmethod
    def _unwrap_exec(parts: list[str]) -> list[str] | None:
        index = 1
        while index < len(parts):
            part = parts[index]
            if part == "--":
                index += 1
                break
            if part == "-a":
                if index + 1 >= len(parts):
                    return None
                index += 2
                continue
            if part.startswith("-") and len(part) > 1:
                if set(part[1:]) <= {"c", "l"}:
                    index += 1
                    continue
                return None
            break
        return parts[index:] or None

    @staticmethod
    def _unwrap_shell_script(parts: list[str]) -> str | None:
        if len(parts) < 3 or parts[1] not in {"-c", "-lc"}:
            return None
        return parts[2]

    @staticmethod
    def _is_assignment(part: str) -> bool:
        return _SHELL_ASSIGNMENT.match(part) is not None

    @staticmethod
    def _contains_expansion_marker(part: str) -> bool:
        return any(marker in part for marker in "$*?[{~")

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
