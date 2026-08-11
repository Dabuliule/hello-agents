from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.core.ids import new_id
from codecraft.eval import (
    EvalTask,
    get_eval_tasks,
    render_eval_html,
    render_eval_json,
    run_eval_suite,
)
from codecraft.schema.session import SessionSource


class EvalFormat(StrEnum):
    """CLI 可写出的评测报告格式组合。"""

    JSON = "json"
    HTML = "html"
    BOTH = "both"


def register_eval_command(app: typer.Typer) -> None:
    """向 Typer 应用注册 ``eval`` 子命令。"""

    @app.command("eval")
    def eval_command(
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider", help="Model provider: openai, qwen, or deepseek."
            ),
        ] = None,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Model name."),
        ] = None,
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        config: Annotated[
            Path | None,
            typer.Option("--config", help="Highest-priority TOML config file."),
        ] = None,
        profile: Annotated[
            str | None,
            typer.Option("--profile", help="Profile name under ~/.codecraft/profiles."),
        ] = None,
        task_ids: Annotated[
            list[str] | None,
            typer.Option(
                "--task",
                help="Task id to run. Repeat to select multiple tasks.",
            ),
        ] = None,
        limit: Annotated[
            int | None,
            typer.Option("--limit", min=1, help="Run only the first N selected tasks."),
        ] = None,
        repeat: Annotated[
            int,
            typer.Option(
                "--repeat",
                min=1,
                max=10,
                help="Run each selected task N times.",
            ),
        ] = 1,
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                "-o",
                help="Directory for the report, workspaces, and task traces.",
            ),
        ] = None,
        format: Annotated[
            EvalFormat,
            typer.Option("--format", help="Evaluation report format."),
        ] = EvalFormat.BOTH,
        list_only: Annotated[
            bool,
            typer.Option(
                "--list", help="List the built-in tasks without running them."
            ),
        ] = False,
    ) -> None:
        """列出内置任务或同步驱动异步 Eval Suite，并映射退出码。"""
        tasks = get_eval_tasks()
        if list_only:
            _print_task_list(tasks)
            return

        import asyncio

        exit_code = asyncio.run(
            run_eval(
                provider=provider,
                model=model,
                codecraft_home=codecraft_home,
                config_path=config,
                profile=profile,
                task_ids=task_ids,
                limit=limit,
                repeat=repeat,
                output_dir=output_dir,
                format=format,
            )
        )
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_eval(
    *,
    provider: str | None,
    model: str | None,
    codecraft_home: Path,
    config_path: Path | None,
    profile: str | None,
    task_ids: list[str] | None,
    limit: int | None,
    repeat: int,
    output_dir: Path | None,
    format: EvalFormat,
) -> int:
    """选择 Task、强制安全配置、运行评测并写 JSON/HTML 报告。

    返回 0 表示全部 attempts 通过，1 表示至少一个失败，2 表示选择/输出目录
    用法错误；每次 attempt 的进度即时写到 Console。
    """
    from codecraft.cli import app as cli_app

    console = make_console()
    try:
        tasks = _select_tasks(get_eval_tasks(), task_ids, limit)
    except ValueError as exc:
        console.print(str(exc), style="error", markup=False)
        return 2

    home = codecraft_home.expanduser().resolve()
    base_config = cli_app._load_session_config(
        source=SessionSource.CLI_EVAL,
        provider=provider,
        model=model,
        codecraft_home=home,
        config_path=config_path,
        profile=profile,
        approval_policy=ApprovalPolicy.NEVER,
        network=False,
    )
    destination = (output_dir or _default_output_dir(home)).expanduser().resolve()
    console.print(
        f"eval_suite: {len(tasks)} task(s) x {repeat} repeat(s) = "
        f"{len(tasks) * repeat} evaluation(s) with "
        f"{base_config.model_provider}/{base_config.model}",
        style="muted",
        soft_wrap=True,
    )

    try:
        report = await run_eval_suite(
            tasks=tasks,
            base_config=base_config,
            llm_providers=cli_app._build_provider_registry(base_config),
            output_dir=destination,
            repeat=repeat,
            on_task_complete=lambda index, total, result: console.print(
                f"eval_task: {index}/{total} {result['task_id']} "
                f"attempt={result['attempt']} "
                f"status={result['status']} duration_ms={result['duration_ms']}",
                style="muted" if result["status"] == "passed" else "error",
                soft_wrap=True,
            ),
        )
    except FileExistsError as exc:
        console.print(str(exc), style="error", markup=False, soft_wrap=True)
        return 2

    written: list[Path] = []
    if format in {EvalFormat.JSON, EvalFormat.BOTH}:
        json_path = destination / "eval-report.json"
        json_path.write_text(render_eval_json(report), encoding="utf-8")
        written.append(json_path)
    if format in {EvalFormat.HTML, EvalFormat.BOTH}:
        html_path = destination / "eval-report.html"
        html_path.write_text(render_eval_html(report), encoding="utf-8")
        written.append(html_path)

    metrics = report["metrics"]
    console.print(
        f"eval_success_rate: {metrics['success_rate']:.1f}% "
        f"({metrics['passed_count']}/{metrics['evaluation_count']})",
        style="muted",
        soft_wrap=True,
    )
    for path in written:
        label = "eval_json" if path.suffix == ".json" else "eval_html"
        console.print(f"{label}: {path}", style="muted", soft_wrap=True)
    return 0 if metrics["failed_count"] == 0 else 1


def _select_tasks(
    tasks: tuple[EvalTask, ...],
    task_ids: list[str] | None,
    limit: int | None,
) -> tuple[EvalTask, ...]:
    """按用户顺序选择唯一已知 Task，再应用可选前 N 项限制。"""
    by_id = {task.task_id: task for task in tasks}
    if task_ids:
        duplicates = sorted(
            task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate eval task(s): {', '.join(duplicates)}")
        unknown = [task_id for task_id in task_ids if task_id not in by_id]
        if unknown:
            raise ValueError(f"Unknown eval task(s): {', '.join(unknown)}")
        selected = tuple(by_id[task_id] for task_id in task_ids)
    else:
        selected = tasks
    return selected[:limit] if limit is not None else selected


def _default_output_dir(home: Path) -> Path:
    """用 UTC 时间与随机短后缀生成不碰撞的默认 eval run 目录。"""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = new_id("")[:8]
    return home / "evals" / f"{timestamp}-{suffix}"


def _print_task_list(tasks: tuple[EvalTask, ...]) -> None:
    """打印稳定 task id、category 和 title，不启动 Provider。"""
    console = make_console()
    for task in tasks:
        console.print(
            f"{task.task_id} category={task.category} title={task.title}",
            style="muted",
            soft_wrap=True,
        )
