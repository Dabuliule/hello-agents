from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.ui import make_console
from codecraft.core.ids import new_id
from codecraft.retrieval import (
    get_retrieval_cases,
    render_retrieval_html,
    render_retrieval_json,
    run_retrieval_benchmark,
)


class RetrievalEvalFormat(StrEnum):
    """Retrieval benchmark 可输出的报告格式。"""

    JSON = "json"
    HTML = "html"
    BOTH = "both"


class RetrievalEvalStrategy(StrEnum):
    """可直接比较的自动、扫描、词法和符号策略。"""

    AUTO = "auto"
    SCAN = "scan"
    LEXICAL = "lexical"
    SYMBOL = "symbol"


def register_retrieval_eval_command(app: typer.Typer) -> None:
    """注册固定语料检索 benchmark 子命令。"""

    @app.command("retrieval-eval")
    def retrieval_eval_command(
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        repeat: Annotated[
            int,
            typer.Option(
                "--repeat",
                min=1,
                max=100,
                help="Run each fixed query N times.",
            ),
        ] = 3,
        strategy: Annotated[
            RetrievalEvalStrategy,
            typer.Option("--strategy", help="Retrieval strategy to benchmark."),
        ] = RetrievalEvalStrategy.SCAN,
        output_dir: Annotated[
            Path | None,
            typer.Option(
                "--output-dir",
                "-o",
                help="Directory for the reports and fixed benchmark workspace.",
            ),
        ] = None,
        format: Annotated[
            RetrievalEvalFormat,
            typer.Option("--format", help="Retrieval report format."),
        ] = RetrievalEvalFormat.BOTH,
        list_only: Annotated[
            bool,
            typer.Option("--list", help="List the fixed queries without running them."),
        ] = False,
    ) -> None:
        """列出 cases 或同步驱动异步 benchmark 并映射退出码。"""
        if list_only:
            _print_case_list()
            return

        home = codecraft_home.expanduser().resolve()
        destination = (output_dir or _default_output_dir(home)).expanduser().resolve()
        exit_code = asyncio.run(
            run_retrieval_eval(
                output_dir=destination,
                repeat=repeat,
                strategy=strategy,
                format=format,
            )
        )
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_retrieval_eval(
    *,
    output_dir: Path,
    repeat: int,
    strategy: RetrievalEvalStrategy,
    format: RetrievalEvalFormat,
) -> int:
    """运行 benchmark、按选择写报告并打印质量和延迟摘要。"""
    console = make_console()
    cases = get_retrieval_cases()
    console.print(
        f"retrieval_suite: {len(cases)} case(s) x {repeat} repeat(s) = "
        f"{len(cases) * repeat} evaluation(s)",
        style="muted",
        soft_wrap=True,
    )
    try:
        report = await run_retrieval_benchmark(
            cases=cases,
            output_dir=output_dir,
            repeat=repeat,
            strategy=strategy,
        )
    except FileExistsError as exc:
        console.print(str(exc), style="error", markup=False, soft_wrap=True)
        return 2

    written: list[Path] = []
    if format in {RetrievalEvalFormat.JSON, RetrievalEvalFormat.BOTH}:
        json_path = output_dir / "retrieval-report.json"
        json_path.write_text(render_retrieval_json(report), encoding="utf-8")
        written.append(json_path)
    if format in {RetrievalEvalFormat.HTML, RetrievalEvalFormat.BOTH}:
        html_path = output_dir / "retrieval-report.html"
        html_path.write_text(render_retrieval_html(report), encoding="utf-8")
        written.append(html_path)

    metrics = report["metrics"]
    console.print(
        f"retrieval_quality: recall@1={metrics['mean_recall_at_1']:.3f} "
        f"recall@5={metrics['mean_recall_at_5']:.3f} "
        f"precision@5={metrics['mean_precision_at_5']:.3f} "
        f"mrr={metrics['mean_reciprocal_rank']:.3f}",
        style="muted",
        soft_wrap=True,
    )
    console.print(
        f"retrieval_latency: p50={metrics['latency_p50_ms']}ms "
        f"p95={metrics['latency_p95_ms']}ms",
        style="muted",
        soft_wrap=True,
    )
    for path in written:
        label = "retrieval_json" if path.suffix == ".json" else "retrieval_html"
        console.print(f"{label}: {path}", style="muted", soft_wrap=True)
    return 0


def _default_output_dir(home: Path) -> Path:
    """用 UTC 时间和随机短后缀生成默认 retrieval-evals 目录。"""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = new_id("")[:8]
    return home / "retrieval-evals" / f"{timestamp}-{suffix}"


def _print_case_list() -> None:
    """打印 case id、category 与 query，不创建 workspace。"""
    console = make_console()
    for case in get_retrieval_cases():
        console.print(
            f"{case.case_id} category={case.category} query={case.query}",
            style="muted",
            soft_wrap=True,
        )
