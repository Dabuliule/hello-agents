from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from codecraft.cli.app import app
from codecraft.retrieval.benchmark import (
    _precision_at_k,
    _recall_at_k,
    _reciprocal_rank,
    _unique_paths,
)
from codecraft.retrieval import (
    get_retrieval_cases,
    render_retrieval_html,
    run_retrieval_benchmark,
    seed_retrieval_workspace,
)

runner = CliRunner()


def test_retrieval_metrics_rank_unique_paths_in_first_seen_order():
    matches = [
        {"path": "src/auth/service.py", "line": 1},
        {"path": "src/auth/service.py", "line": 8},
        {"path": "src/unrelated.py", "line": 2},
        {"path": 42, "line": 3},
        "invalid",
    ]

    retrieved = _unique_paths(matches)
    relevant = {"src/auth/service.py", "src/auth/permissions.py"}

    assert retrieved == ["src/auth/service.py", "src/unrelated.py"]
    assert _recall_at_k(retrieved, relevant, 1) == 0.5
    assert _reciprocal_rank(retrieved, relevant) == 1.0
    # 当前报告使用“实际可见结果”作分母：Top-5 只返回两条时精度为 1/2。
    assert _precision_at_k(retrieved, relevant, 5) == 0.5


def test_retrieval_suite_has_stable_multi_language_cases(tmp_path):
    cases = get_retrieval_cases()
    seed_retrieval_workspace(tmp_path)

    assert len(cases) == 10
    assert len({case.case_id for case in cases}) == 10
    assert {case.category for case in cases} >= {
        "exact",
        "path",
        "multi_file",
        "semantic",
    }
    assert (tmp_path / "src/auth/service.py").is_file()
    assert (tmp_path / "src/queue/worker.ts").is_file()
    assert (tmp_path / "src/db/pool.go").is_file()


def test_retrieval_benchmark_reports_quality_cost_and_latency(tmp_path):
    output_dir = tmp_path / "retrieval-run"

    report = asyncio.run(
        run_retrieval_benchmark(
            cases=get_retrieval_cases(),
            output_dir=output_dir,
            repeat=2,
        )
    )

    assert report["schema_version"] == 1
    assert report["run"]["retriever"] == "workspace_search_scan"
    assert report["run"]["case_count"] == 10
    assert report["run"]["evaluation_count"] == 20
    metrics = report["metrics"]
    assert metrics["mean_recall_at_1"] == 0.75
    assert metrics["mean_recall_at_5"] == 0.8
    assert metrics["mean_precision_at_5"] == 0.8
    assert metrics["mean_reciprocal_rank"] == 0.8
    assert metrics["zero_result_count"] == 4
    assert metrics["latency_p50_ms"] >= 0
    assert metrics["latency_p95_ms"] >= metrics["latency_p50_ms"]
    assert metrics["mean_scanned_files"] > 0
    assert metrics["total_scanned_bytes"] > 0
    assert metrics["mean_returned_chars"] > 0
    assert metrics["mean_estimated_returned_tokens"] > 0
    semantic = [case for case in report["cases"] if case["category"] == "semantic"]
    assert len(semantic) == 2
    assert all(case["mean_recall_at_5"] == 0 for case in semantic)
    assert "Recall@5" in render_retrieval_html(report)


def test_lexical_retrieval_benchmark_improves_natural_language_recall(tmp_path):
    report = asyncio.run(
        run_retrieval_benchmark(
            cases=get_retrieval_cases(),
            output_dir=tmp_path / "lexical-run",
            repeat=1,
            strategy="lexical",
        )
    )

    assert report["run"]["retriever"] == "workspace_search_lexical"
    assert report["metrics"]["mean_recall_at_5"] > 0.8
    permissions = next(
        case
        for case in report["cases"]
        if case["case_id"] == "natural-language-permissions"
    )
    assert permissions["mean_recall_at_5"] == 1.0


def test_auto_retrieval_preserves_quality_while_reducing_scan_work(tmp_path):
    report = asyncio.run(
        run_retrieval_benchmark(
            cases=get_retrieval_cases(),
            output_dir=tmp_path / "auto-run",
            repeat=1,
            strategy="auto",
        )
    )

    metrics = report["metrics"]
    assert report["run"]["retriever"] == "workspace_search_auto"
    assert metrics["mean_recall_at_5"] >= 0.9
    assert metrics["mean_scanned_files"] < 4
    assert metrics["retriever_counts"] == {
        "lexical": 6,
        "scan": 2,
        "symbol": 2,
    }
    assert metrics["mean_precision_at_5"] == 0.85
    assert metrics["irrelevant_path_count"] == 2
    assert metrics["mean_retriever_attempts"] > 1
    routed = {result["case_id"]: result for result in report["results"]}
    assert routed["exact-symbol"]["attempted_retrievers"] == ["symbol"]
    assert routed["exact-symbol"]["retrieved_paths"] == ["src/auth/service.py"]
    assert routed["natural-language-permissions"]["retrieved_paths"][0] == (
        "src/auth/permissions.py"
    )
    assert routed["natural-language-reconnect"]["retriever"] == "lexical"
    assert routed["natural-language-reconnect"]["recall_at_5"] == 0.0


def test_retrieval_eval_command_writes_reports_without_model_calls(tmp_path):
    output_dir = tmp_path / "retrieval-cli"

    listed = runner.invoke(app, ["retrieval-eval", "--list"])
    result = runner.invoke(
        app,
        [
            "retrieval-eval",
            "--repeat",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert listed.exit_code == 0
    assert "natural-language-permissions" in listed.output
    assert result.exit_code == 0, result.output
    assert (
        "retrieval_quality: recall@1=0.750 recall@5=0.800 precision@5=0.800 mrr=0.800"
    ) in result.output
    json_path = output_dir / "retrieval-report.json"
    html_path = output_dir / "retrieval-report.html"
    assert json_path.is_file()
    assert html_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["metrics"]["evaluation_count"] == 10

    repeated = runner.invoke(
        app,
        [
            "retrieval-eval",
            "--repeat",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert repeated.exit_code == 2
    assert "already contains run artifacts" in repeated.output
    assert "Traceback" not in repeated.output
