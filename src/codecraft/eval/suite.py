from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

EVAL_SUITE_NAME = "codecraft-core-v1"


class EvalCheckType(StrEnum):
    """内置 grader 支持的精确文本、包含关系和 JSON 字段断言。"""

    FILE_EQUALS = "file_equals"
    FILE_CONTAINS = "file_contains"
    FILE_NOT_CONTAINS = "file_not_contains"
    JSON_EQUALS = "json_equals"


@dataclass(frozen=True)
class EvalCheck:
    """一个 workspace 相对路径上的预期值与可选 JSON dotted path。"""

    kind: EvalCheckType
    path: str
    expected: Any
    json_path: str | None = None


@dataclass(frozen=True)
class EvalTask:
    """稳定任务 ID、类别、Prompt、初始文件和全部确定性检查。

    Task 同时声明输入 fixture 与客观 outcome contract，不使用模型最终回答或另一个
    LLM judge 判分。checks 应同时覆盖目标变化和关键不变量；例如更新版本号时也检查
    README 未被误改，避免“完成主目标但产生无关破坏”被计为通过。
    """

    task_id: str
    title: str
    category: str
    prompt: str
    seed_files: dict[str, str]
    checks: tuple[EvalCheck, ...]


def get_eval_tasks() -> tuple[EvalTask, ...]:
    """返回覆盖创建、定点/多文件编辑、检索、指令遵循和重构的固定套件。"""
    return (
        EvalTask(
            task_id="create-welcome-file",
            title="Create an exact file",
            category="file_creation",
            prompt=(
                "Create a file named welcome.txt containing exactly "
                "`hello, codecraft` followed by a newline. Do not change other files."
            ),
            seed_files={"README.md": "# Tiny workspace\n"},
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "welcome.txt",
                    "hello, codecraft\n",
                ),
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "README.md",
                    "# Tiny workspace\n",
                ),
            ),
        ),
        EvalTask(
            task_id="fix-calculator-bug",
            title="Make a minimal bug fix",
            category="targeted_edit",
            prompt=(
                "Fix the bug in calculator.py so add(2, 3) returns 5. Preserve the "
                "subtract function and the module comment exactly."
            ),
            seed_files={
                "calculator.py": (
                    "# Arithmetic helpers used by the invoice service.\n\n"
                    "def add(left, right):\n"
                    "    return left - right\n\n\n"
                    "def subtract(left, right):\n"
                    "    return left - right\n"
                )
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "calculator.py",
                    "# Arithmetic helpers used by the invoice service.\n\n"
                    "def add(left, right):\n"
                    "    return left + right\n\n\n"
                    "def subtract(left, right):\n"
                    "    return left - right\n",
                ),
            ),
        ),
        EvalTask(
            task_id="sync-package-version",
            title="Synchronize a multi-file change",
            category="multi_file_edit",
            prompt=(
                "Update the package version from 1.4.2 to 1.5.0 everywhere it is "
                "declared. Keep all unrelated text unchanged."
            ),
            seed_files={
                "pyproject.toml": ('[project]\nname = "tiny-app"\nversion = "1.4.2"\n'),
                "src/tiny_app/__init__.py": '__version__ = "1.4.2"\n',
                "README.md": "Tiny App supports protocol 1.4.2.\n",
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "pyproject.toml",
                    '[project]\nname = "tiny-app"\nversion = "1.5.0"\n',
                ),
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "src/tiny_app/__init__.py",
                    '__version__ = "1.5.0"\n',
                ),
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "README.md",
                    "Tiny App supports protocol 1.4.2.\n",
                ),
            ),
        ),
        EvalTask(
            task_id="write-quickstart",
            title="Create a nested documentation file",
            category="documentation",
            prompt=(
                "Create docs/quickstart.md. It must have the heading `# Quickstart` "
                "and mention both commands `uv sync` and `uv run tiny-app`."
            ),
            seed_files={"docs/.keep": "", "README.md": "# Tiny App\n"},
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "docs/quickstart.md",
                    "# Quickstart",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "docs/quickstart.md",
                    "uv sync",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "docs/quickstart.md",
                    "uv run tiny-app",
                ),
            ),
        ),
        EvalTask(
            task_id="update-json-settings",
            title="Edit structured configuration",
            category="structured_data",
            prompt=(
                "In settings.json, enable features.search and change retries to 3. "
                "Preserve the service name and keep the file valid JSON."
            ),
            seed_files={
                "settings.json": (
                    "{\n"
                    '  "service": "catalog",\n'
                    '  "features": {"search": false, "export": true},\n'
                    '  "retries": 2\n'
                    "}\n"
                )
            },
            checks=(
                EvalCheck(
                    EvalCheckType.JSON_EQUALS,
                    "settings.json",
                    True,
                    "features.search",
                ),
                EvalCheck(
                    EvalCheckType.JSON_EQUALS,
                    "settings.json",
                    3,
                    "retries",
                ),
                EvalCheck(
                    EvalCheckType.JSON_EQUALS,
                    "settings.json",
                    "catalog",
                    "service",
                ),
                EvalCheck(
                    EvalCheckType.JSON_EQUALS,
                    "settings.json",
                    True,
                    "features.export",
                ),
            ),
        ),
        EvalTask(
            task_id="locate-legacy-token",
            title="Retrieve repository context",
            category="repository_search",
            prompt=(
                "Find the file containing the exact text LEGACY_TOKEN. Create "
                "migration.txt containing only that file's relative path and a newline."
            ),
            seed_files={
                "src/current/auth.py": 'TOKEN_KIND = "CURRENT"\n',
                "src/legacy/auth.py": 'TOKEN_KIND = "LEGACY_TOKEN"\n',
                "docs/auth.md": "Authentication notes.\n",
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "migration.txt",
                    "src/legacy/auth.py\n",
                ),
            ),
        ),
        EvalTask(
            task_id="edit-production-timeout",
            title="Change the correct configuration section",
            category="contextual_edit",
            prompt=(
                "Change only the production timeout in config.ini from 20 to 45. "
                "Leave the development timeout and all comments unchanged."
            ),
            seed_files={
                "config.ini": (
                    "# Request settings\n"
                    "[development]\n"
                    "timeout = 20\n\n"
                    "[production]\n"
                    "timeout = 20\n"
                )
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "config.ini",
                    "# Request settings\n"
                    "[development]\n"
                    "timeout = 20\n\n"
                    "[production]\n"
                    "timeout = 45\n",
                ),
            ),
        ),
        EvalTask(
            task_id="deduplicate-timeout-constant",
            title="Refactor a shared constant",
            category="refactoring",
            prompt=(
                "Define DEFAULT_TIMEOUT = 30 in constants.py. Update api.py and "
                "worker.py to import and use DEFAULT_TIMEOUT instead of duplicating 30."
            ),
            seed_files={
                "constants.py": 'APP_NAME = "tiny-app"\n',
                "api.py": "timeout = 30\n",
                "worker.py": "timeout = 30\n",
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "constants.py",
                    "DEFAULT_TIMEOUT = 30",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "api.py",
                    "from constants import DEFAULT_TIMEOUT",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "api.py",
                    "timeout = DEFAULT_TIMEOUT",
                ),
                EvalCheck(
                    EvalCheckType.FILE_NOT_CONTAINS,
                    "api.py",
                    "timeout = 30",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "worker.py",
                    "from constants import DEFAULT_TIMEOUT",
                ),
                EvalCheck(
                    EvalCheckType.FILE_CONTAINS,
                    "worker.py",
                    "timeout = DEFAULT_TIMEOUT",
                ),
                EvalCheck(
                    EvalCheckType.FILE_NOT_CONTAINS,
                    "worker.py",
                    "timeout = 30",
                ),
            ),
        ),
        EvalTask(
            task_id="follow-project-instructions",
            title="Respect project instructions",
            category="instruction_following",
            prompt=("Mark the project ready according to the repository instructions."),
            seed_files={
                "AGENTS.md": (
                    "Do not modify locked.txt. To mark the project ready, append "
                    "the line `status=ready` to state.txt. Preserve existing lines.\n"
                ),
                "locked.txt": "owner=platform\n",
                "state.txt": "name=tiny-app\n",
            },
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "state.txt",
                    "name=tiny-app\nstatus=ready\n",
                ),
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "locked.txt",
                    "owner=platform\n",
                ),
            ),
        ),
        EvalTask(
            task_id="normalize-name-list",
            title="Apply constrained text transformation",
            category="data_cleanup",
            prompt=(
                "In names.txt, keep the first comment line unchanged, then sort the "
                "names alphabetically and remove duplicates."
            ),
            seed_files={"names.txt": "# Active users\nzoe\nalice\nbob\nalice\n"},
            checks=(
                EvalCheck(
                    EvalCheckType.FILE_EQUALS,
                    "names.txt",
                    "# Active users\nalice\nbob\nzoe\n",
                ),
            ),
        ),
    )


def seed_workspace(task: EvalTask, workspace: Path) -> None:
    """在独立 workspace 创建任务声明的全部 UTF-8 初始文件。"""
    workspace.mkdir(parents=True, exist_ok=True)
    for relative, content in task.seed_files.items():
        path = _workspace_path(workspace, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def evaluate_task(task: EvalTask, workspace: Path) -> list[dict[str, Any]]:
    """按任务声明顺序执行所有检查并返回可序列化 grader 结果。"""
    return [_evaluate_check(check, workspace) for check in task.checks]


def _evaluate_check(check: EvalCheck, workspace: Path) -> dict[str, Any]:
    """读取真实文件执行一个 check；缺失、编码、JSON/path 错误均判失败。

    模型最终回答不参与 correctness 判定；actual 文本最多保留 500 字符，避免
    失败报告被整个大文件占满。
    """
    path = _workspace_path(workspace, check.path)
    actual: Any = None
    error: str | None = None

    try:
        content = path.read_text(encoding="utf-8")
        if check.kind == EvalCheckType.FILE_EQUALS:
            actual = content
            passed = content == check.expected
        elif check.kind == EvalCheckType.FILE_CONTAINS:
            actual = content
            passed = str(check.expected) in content
        elif check.kind == EvalCheckType.FILE_NOT_CONTAINS:
            actual = content
            passed = str(check.expected) not in content
        elif check.kind == EvalCheckType.JSON_EQUALS:
            data = json.loads(content)
            actual = _json_value(data, check.json_path or "")
            passed = actual == check.expected
        else:
            raise ValueError(f"unsupported eval check: {check.kind}")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        passed = False
        error = str(exc)

    return {
        "kind": check.kind.value,
        "path": check.path,
        "json_path": check.json_path,
        "passed": passed,
        "expected": check.expected,
        "actual": _preview(actual),
        "error": error,
    }


def _workspace_path(workspace: Path, relative: str) -> Path:
    """将 POSIX 相对路径映射到 workspace，并拒绝绝对路径或 ``..`` 逃逸。"""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"eval path must be workspace-relative: {relative}")
    return workspace.joinpath(*pure.parts)


def _json_value(data: Any, path: str) -> Any:
    """沿点分 object key 读取 JSON 值，非 object 中间节点明确报错。"""
    value = data
    for part in path.split(".") if path else ():
        if not isinstance(value, dict):
            raise TypeError(f"{part} is not inside an object")
        value = value[part]
    return value


def _preview(value: Any, limit: int = 500) -> Any:
    """只截断过长字符串 actual；数字、布尔、对象保持原结构。"""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"
