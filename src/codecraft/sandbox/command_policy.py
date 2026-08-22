from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import posixpath
import re
import shlex

from pydantic import BaseModel


class CommandRisk(StrEnum):
    """无需审批、必须审批和绝不执行三档静态命令风险。"""

    SAFE = "safe"
    PROMPT = "prompt"
    DENY = "deny"


class CommandDecision(BaseModel):
    """命令判级、可解释原因和是否进入用户审批的稳定结果。

    ``risk`` 用于审计和 BashTool 的最终硬拒绝；``requires_approval`` 直接驱动
    ApprovalManager。当前 SAFE/DENY 都不弹窗，前者可继续、后者由工具拒绝；只有
    PROMPT 请求用户决定，避免把“是否询问”和“是否允许”混成一个布尔值。
    """

    risk: CommandRisk
    reason: str
    requires_approval: bool


@dataclass(frozen=True)
class _ShellScan:
    """轻量 shell 扫描得到的命令段、控制运算符和动态求值标志。"""

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
    """用小型只读白名单和 fail-closed 规则静态判定 shell 命令风险。

    只有完整 argv 命中白名单才是 SAFE；未知命令、未知参数、控制运算符和普通变量/
    glob 展开至少 PROMPT；命令替换、进程替换、无法可靠解包的 wrapper 和宽泛
    ``rm -rf`` 直接 DENY。策略只做保守静态判级，不执行 shell，也不声称理解全部
    shell 语义；真正的文件、网络和进程边界仍由 SandboxBackend 强制。

    判级采用白名单而不是“危险词黑名单”：同一可执行文件的 flag、重定向、wrapper
    或后续 segment 都可能改变副作用，只有核对完整形状才能给出无需审批的 SAFE。
    """

    def classify(
        self, command: str, *, network_access: bool = False
    ) -> CommandDecision:
        """返回整条 shell 文本的最严格风险、原因和审批要求。

        Args:
            command: 模型请求执行的原始 shell 文本，不在判级期间展开或执行。
            network_access: Runtime 是否具备网络能力；具备能力仍不等于免审批。

        Returns:
            SAFE、PROMPT 或 DENY 的稳定 ``CommandDecision``。网络命令在能力关闭时
            DENY，能力开启时仍为 PROMPT；用户批准也不能把 DENY 变成允许。

        Example:
            >>> CommandPolicy().classify("pwd").risk
            <CommandRisk.SAFE: 'safe'>
            >>> CommandPolicy().classify("sudo pwd").risk
            <CommandRisk.DENY: 'deny'>
        """
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
        """扫描整条 shell 文本并聚合每个 segment 的最严格决定。

        命令/进程替换直接 DENY；控制运算符和普通 expansion 至少 PROMPT；
        任一 segment DENY 会压过其他结果。聚合顺序固定为
        ``substitution → segment DENY → expanded rm -rf → operator → expansion``；
        只有单段且完整命中只读白名单才 SAFE。
        """
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
        """按 wrapper、破坏性、网络、精确白名单顺序判定一个 argv。

        wrapper 必须先解包，否则 ``env sudo``、``sh -c 'rm -rf /'`` 会只看到外层
        可执行文件；硬拒绝先于网络和白名单，默认分支则保守落到 PROMPT。
        """
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
        """识别赋值、env/command/exec 和 shell -c 间接调用并递归判级。

        能可靠还原实际 argv/脚本文本时继续递归，不能还原时直接 DENY；深度上限
        防止恶意嵌套消耗和利用解析差异绕过内层危险命令检查。
        """
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
        """递归判定解包后的 argv；任何 wrapper 都至少把 SAFE 提升为 PROMPT。"""
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
        """保留内层 DENY，否则因间接执行把 SAFE/PROMPT 统一提升为 PROMPT。

        即使 ``env LANG=C pwd`` 的内层 argv 是只读白名单，wrapper 仍会改变环境、
        PATH 查找或 shell 启动语义，因此不继承 SAFE。
        """
        if decision.risk == CommandRisk.DENY:
            return decision
        return CommandDecision(
            risk=CommandRisk.PROMPT,
            reason=f"{wrapper} wrapper requires approval",
            requires_approval=True,
        )

    @staticmethod
    def _opaque_wrapper_decision(reason: str) -> CommandDecision:
        """对无法可靠还原实际命令的 wrapper 采用 fail-closed DENY。"""
        return CommandDecision(
            risk=CommandRisk.DENY,
            reason=reason,
            requires_approval=False,
        )

    @staticmethod
    def _wrapper_depth_decision() -> CommandDecision:
        """拒绝超过八层的 wrapper，防止递归消耗和策略绕过。"""
        return CommandDecision(
            risk=CommandRisk.DENY,
            reason=f"command wrapper nesting exceeds {_MAX_WRAPPER_DEPTH} levels",
            requires_approval=False,
        )

    @staticmethod
    def _leading_assignment_count(parts: list[str]) -> int:
        """统计 argv 前部连续的 ``NAME=value`` 环境赋值。"""
        index = 0
        while index < len(parts) and _SHELL_ASSIGNMENT.match(parts[index]):
            index += 1
        return index

    @classmethod
    def _unwrap_command(cls, executable: str, parts: list[str]) -> list[str] | None:
        """按 env、command 或 exec 语法分发 wrapper 解包。"""
        if executable == "env":
            return cls._unwrap_env(parts)
        if executable == "command":
            return cls._unwrap_command_builtin(parts)
        return cls._unwrap_exec(parts)

    @classmethod
    def _unwrap_env(cls, parts: list[str]) -> list[str] | None:
        """跳过受支持 env 选项和赋值，返回实际命令；不支持 -S。"""
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
        """返回受支持 env 选项后的索引，缺值或不透明选项返回 None。"""
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
        """处理 command 的 -p/--，返回实际 argv。"""
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
        """处理 exec 的 -a、-c、-l 和 --，返回实际 argv。"""
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
        """只接受 shell ``-c``/``-lc`` 并返回其脚本文本。"""
        if len(parts) < 3 or parts[1] not in {"-c", "-lc"}:
            return None
        return parts[2]

    @staticmethod
    def _is_assignment(part: str) -> bool:
        """判断一个 argv 是否以合法环境变量赋值开头。"""
        return _SHELL_ASSIGNMENT.match(part) is not None

    @staticmethod
    def _contains_expansion_marker(part: str) -> bool:
        """保守识别变量、glob、brace 和 home expansion 标记。"""
        return any(marker in part for marker in "$*?[{~")

    @staticmethod
    def _split(command: str) -> list[str]:
        """用 shlex 拆 argv；引号不闭合等无效语法返回空列表。"""
        try:
            return shlex.split(command)
        except ValueError:
            return []

    @staticmethod
    def _is_exact_read_only(parts: list[str]) -> bool:
        """完整匹配固定命令或受限 ls/rg 语法，未知参数不放行。"""
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
        """只允许无参数或由 aAlh1Fp 组成的短 ls 展示 flags。"""
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
        """只允许 --files、单 pattern 或一个白名单 flag 加 pattern。"""
        if arguments == ["--files"]:
            return True
        if len(arguments) == 1:
            return bool(arguments[0]) and not arguments[0].startswith("-")
        if len(arguments) == 2 and arguments[0] in _READ_ONLY_RG_FLAGS:
            return bool(arguments[1]) and not arguments[1].startswith("-")
        return False

    @staticmethod
    def _is_destructive_rm(parts: list[str]) -> bool:
        """识别 recursive+force 且目标为根、当前目录、父目录或宽泛 glob。"""
        if not CommandPolicy._is_recursive_force_rm(parts):
            return False

        return any(
            posixpath.normpath(target) in _BROAD_RM_TARGETS
            for target in CommandPolicy._rm_targets(parts[1:])
        )

    @staticmethod
    def _is_recursive_force_rm(parts: list[str]) -> bool:
        """识别 rm argv 是否同时包含递归与强制 flags。"""
        if not parts or CommandPolicy._command_name(parts[0]) != "rm":
            return False
        has_recursive, has_force = CommandPolicy._rm_flags(parts[1:])
        return has_recursive and has_force

    @staticmethod
    def _rm_flags(arguments: list[str]) -> tuple[bool, bool]:
        """解析 rm 长短及合并 flags，返回 recursive/force 两个布尔值。"""
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
        """跳过 -- 前选项并返回 rm 目标参数。"""
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
        """越过带值/无值的 Git 全局选项，定位真实 subcommand。"""
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
        """同时兼容 POSIX/Windows 分隔符提取可执行文件 basename。"""
        return executable.replace("\\", "/").rsplit("/", 1)[-1]


class _ShellScanner:
    """不执行 shell 的单遍词法扫描器，用于发现控制语法和动态求值。

    Scanner 保留单/双引号与转义语境，识别真正生效的 ``$``、反引号、glob、管道、
    重定向和换行，再把各命令段交给 shlex 拆 argv。不能只使用 shlex：去掉引号后，
    单引号中的字面量 ``'$HOME'`` 与双引号中的展开 ``"$HOME"`` 将难以区分。
    这只是保守词法器而非完整 shell parser；无法证明安全的形状不会进入 SAFE。
    """

    def __init__(self, command: str) -> None:
        """初始化引号、转义、当前位置和累计结果状态。"""
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
        """逐字符扫描并返回唯一运算符、非空 segments 与动态标志。"""
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
        """把转义后的字符视为字面量并退出 escaped 状态。"""
        self.current.append(character)
        self.escaped = False
        self.index += 1

    def _consume_quoted(self, character: str, following: str) -> None:
        """在引号内保留文本，并只在双引号内识别替换/变量展开。"""
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
        """在普通状态分派引号、替换、glob、换行和 shell 运算符。"""
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
        """区分 ``$(`` 命令替换与普通 ``$VAR`` expansion。"""
        if following == "(":
            self.has_substitution = True
        else:
            self.has_expansion = True
        self.current.append("$")
        self.index += 1

    def _consume_newline(self, character: str, following: str) -> None:
        """结束当前 segment，并把 CRLF 或单换行记为控制运算符。"""
        self._flush_segment()
        self._add_operator("newline")
        if character == "\r" and following == "\n":
            self.index += 1
        self.index += 1

    def _consume_operator(self, character: str, following: str) -> None:
        """结束 segment，并优先识别双字符 shell 运算符。"""
        self._flush_segment()
        pair = f"{character}{following}"
        if pair in _DOUBLE_SHELL_OPERATORS:
            self._add_operator(pair)
            self.index += 2
        else:
            self._add_operator(character)
            self.index += 1

    def _flush_segment(self) -> None:
        """去空白后保存当前非空 segment 并清空 buffer。"""
        segment = "".join(self.current).strip()
        if segment:
            self.segments.append(segment)
        self.current.clear()

    def _add_operator(self, operator: str) -> None:
        """按首次出现顺序记录唯一控制运算符。"""
        if operator not in self.operators:
            self.operators.append(operator)


def _scan_shell(command: str) -> _ShellScan:
    """扫描 shell 文本，不做变量展开、命令替换或实际执行。

    Example:
        >>> scan = _scan_shell("pwd && rg TODO")
        >>> scan.segments, scan.operators
        (('pwd', 'rg TODO'), ('&&',))
    """
    return _ShellScanner(command).scan()
