from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from time import monotonic, perf_counter_ns
from typing import Any, Protocol

from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.ids import new_id
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox import SandboxMode
from codecraft.schema.tool import ToolCall
from codecraft.retrieval.suite import (
    RETRIEVAL_SUITE_NAME,
    RetrievalCase,
    seed_retrieval_workspace,
)


class SearchTool(Protocol):
    """Benchmark 所需的最小 workspace_search 工具结构协议。"""

    name: str
    args_schema: Any

    async def arun(self, args: Any, context: Any) -> Any:
        """以已校验参数和只读 ToolContext 执行一次公开搜索接口。"""
        ...


RETRIEVAL_REPORT_SCHEMA_VERSION = 1


async def run_retrieval_benchmark(
    *,
    cases: Sequence[RetrievalCase],
    output_dir: Path,
    repeat: int = 3,
    strategy: str = "scan",
    tool: SearchTool | None = None,
    on_case_complete: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """用公开 workspace_search 工具重复运行固定语料并汇总质量/成本。

    这里刻意经过 ``WorkspaceSearchTool.arun``，而不是直接调用某个 Retriever，
    使参数校验、WorkspaceGuard、路由/降级和 ToolResult 格式也进入评测范围。
    整条路径不创建 AgentRuntime、不加载模型，也不消耗模型 API Token。

    所有 case/repeat 顺序读取同一个不可变 workspace。repeat 主要增加延迟样本，
    对确定性检索器重复相同 query 不会产生新的相关性证据，也不能当作独立的
    模型成功率试验；文件系统页缓存还可能让后续样本比首次调用更热。

    Args:
        cases: 唯一 case_id 的确定性评测用例。
        output_dir: 必须没有既有 run artifacts 的新运行目录。
        repeat: 每个用例重复次数，用于观测延迟分布。
        strategy: 传给工具的 ``scan``、``auto`` 等策略。
        tool: 可选替身；省略时构造真实 Scan 或 RepositoryIndex 工具链。
        on_case_complete: 每次评估完成后的进度回调。

    Returns:
        含 schema/run/metrics/cases/results 的可序列化报告。

    Raises:
        ValueError: repeat 非正或 case_id 重复。
        FileExistsError: 输出目录已有本评测保留产物。
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("retrieval cases must have unique case ids")

    output_dir = output_dir.expanduser().resolve()
    _ensure_new_run_directory(output_dir)
    workspace = output_dir / "workspace"
    seed_retrieval_workspace(workspace)

    if tool is None:
        from codecraft.retrieval.engine import ContextEngine
        from codecraft.retrieval.index import RepositoryIndex
        from codecraft.retrieval.retrievers import (
            LexicalRetriever,
            ScanRetriever,
            SymbolRetriever,
        )
        from codecraft.tool.builtin.filesystem import WorkspaceSearchTool

        if strategy == "scan":
            search_tool: SearchTool = WorkspaceSearchTool()
        else:
            repository_index = RepositoryIndex(output_dir / "indexes")
            repository_index.sync(workspace)
            search_tool = WorkspaceSearchTool(
                ContextEngine(
                    [
                        ScanRetriever(),
                        LexicalRetriever(repository_index),
                        SymbolRetriever(repository_index),
                    ]
                )
            )
    else:
        search_tool = tool
    started_at = datetime.now(UTC)
    started = monotonic()
    schedule = [(case, attempt) for case in cases for attempt in range(1, repeat + 1)]
    results: list[dict[str, Any]] = []
    for case_number, (case, attempt) in enumerate(schedule, start=1):
        result = await _run_case(
            search_tool,
            case,
            attempt,
            workspace,
            strategy=strategy,
        )
        results.append(result)
        if on_case_complete is not None:
            on_case_complete(case_number, len(schedule), result)

    finished_at = datetime.now(UTC)
    return {
        "schema_version": RETRIEVAL_REPORT_SCHEMA_VERSION,
        "run": {
            "run_id": new_id("retrieval_eval_"),
            "suite": RETRIEVAL_SUITE_NAME,
            "retriever": f"workspace_search_{strategy}",
            "case_count": len(cases),
            "repeat": repeat,
            "evaluation_count": len(results),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": int((monotonic() - started) * 1000),
            "workspace": str(workspace),
            "output_dir": str(output_dir),
        },
        "metrics": _aggregate_metrics(results),
        "cases": [_case_summary(case, results) for case in cases],
        "results": results,
    }


async def _run_case(
    tool: SearchTool,
    case: RetrievalCase,
    attempt: int,
    workspace: Path,
    *,
    strategy: str,
) -> dict[str, Any]:
    """执行单次 case，按唯一文件路径计算排名并记录工具成本。

    一个文件可能返回多条行级 snippet；排名指标先按首次出现顺序折叠路径，
    防止同一文件的多次文本命中挤占 Top-K 并虚增相关文件数量。
    """
    call = ToolCall(
        call_id=new_id("call_retrieval_"),
        name=tool.name,
        arguments={
            "query": case.query,
            "path": case.path,
            "mode": case.mode,
            "max_results": 10,
            "strategy": strategy,
        },
    )
    args = tool.args_schema.model_validate(call.arguments)
    started = perf_counter_ns()
    result = await tool.arun(args, _tool_context(workspace, call))
    latency_ms = round((perf_counter_ns() - started) / 1_000_000, 3)
    if not result.success or result.data is None:
        raise RuntimeError(result.error or "workspace search failed without an error")

    matches = result.data.get("matches", [])
    retrieved_paths = _unique_paths(matches)
    relevant = set(case.relevant_paths)
    return {
        "case_id": case.case_id,
        "attempt": attempt,
        "category": case.category,
        "query": case.query,
        "path": case.path,
        "mode": case.mode,
        "relevant_paths": list(case.relevant_paths),
        "retrieved_paths": retrieved_paths,
        "recall_at_1": _recall_at_k(retrieved_paths, relevant, 1),
        "recall_at_5": _recall_at_k(retrieved_paths, relevant, 5),
        "precision_at_5": _precision_at_k(retrieved_paths, relevant, 5),
        "reciprocal_rank": _reciprocal_rank(retrieved_paths, relevant),
        "irrelevant_path_count": len(set(retrieved_paths[:5]) - relevant),
        "latency_ms": latency_ms,
        "candidate_file_count": int(result.metadata.get("candidate_file_count", 0)),
        "scanned_file_count": int(result.metadata.get("scanned_file_count", 0)),
        "read_file_count": int(result.metadata.get("read_file_count", 0)),
        "scanned_bytes": int(result.metadata.get("scanned_bytes", 0)),
        "returned_chars": int(result.metadata.get("returned_chars", 0)),
        "estimated_returned_tokens": ceil(
            int(result.metadata.get("returned_chars", 0)) / 4
        ),
        "match_count": int(result.metadata.get("match_count", 0)),
        "retriever": result.metadata.get("retriever"),
        "fallback_from": result.metadata.get("fallback_from"),
        "route_reason": result.metadata.get("route_reason"),
        "attempted_retrievers": result.metadata.get("attempted_retrievers", []),
    }


def _tool_context(workspace: Path, call: ToolCall) -> Any:
    """构造禁止审批、只读、无网络且仅允许一次工具调用的评测上下文。"""
    from codecraft.tool.base import ToolContext

    now = datetime.now(UTC)
    context = TurnContext(
        session_id=new_id("ses_retrieval_"),
        turn_id=new_id("turn_retrieval_"),
        cwd=workspace,
        model="none",
        model_provider="benchmark",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.READ_ONLY,
        network_access=False,
        available_tools=[],
        max_tool_calls=1,
        max_tool_output_chars=80_000,
        created_at=now,
    )
    return ToolContext(context=context, call=call)


def _unique_paths(matches: object) -> list[str]:
    """从不可信工具数据中按首次出现顺序提取唯一合法路径。"""
    if not isinstance(matches, list):
        return []
    paths: list[str] = []
    for match in matches:
        if not isinstance(match, dict) or not isinstance(match.get("path"), str):
            continue
        path = match["path"]
        if path not in paths:
            paths.append(path)
    return paths


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """计算前 k 条结果覆盖的相关路径比例。"""
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """返回首个相关结果排名的倒数，完全未命中返回 0。"""
    for rank, path in enumerate(retrieved, start=1):
        if path in relevant:
            return 1 / rank
    return 0.0


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """计算实际返回的前 k 条中相关路径比例。

    分母是 ``min(len(retrieved), k)``，结果不足 k 条时不补无关项。因此这是
    “可见结果精度”，数值不能直接和固定使用 ``relevant / k`` 的严格 P@K
    benchmark 横向比较；报告同时记录 zero-result 和 irrelevant-path 数辅助解读。
    """
    visible = retrieved[:k]
    if not visible:
        return 0.0
    return len(relevant.intersection(visible)) / len(visible)


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总所有重复结果的质量、最近秩、延迟、资源和路由分布。

    ``mean_*`` 和 percentile 可用于不同 repeat 数的 run 间比较；zero-result、
    irrelevant-path、total-scanned-bytes 等总量会随 repeat 线性增长，比较时必须
    同时查看 ``evaluation_count``，不能只看绝对值。
    """
    count = len(results)
    latencies = [float(result["latency_ms"]) for result in results]
    return {
        "mean_recall_at_1": _mean(results, "recall_at_1"),
        "mean_recall_at_5": _mean(results, "recall_at_5"),
        "mean_precision_at_5": _mean(results, "precision_at_5"),
        "mean_reciprocal_rank": _mean(results, "reciprocal_rank"),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "mean_scanned_files": _mean(results, "scanned_file_count"),
        "total_scanned_bytes": sum(result["scanned_bytes"] for result in results),
        "mean_returned_chars": _mean(results, "returned_chars"),
        "mean_estimated_returned_tokens": _mean(results, "estimated_returned_tokens"),
        "zero_result_count": sum(result["match_count"] == 0 for result in results),
        "irrelevant_path_count": sum(
            result["irrelevant_path_count"] for result in results
        ),
        "retriever_counts": dict(
            sorted(Counter(result["retriever"] for result in results).items())
        ),
        "route_reason_counts": dict(
            sorted(
                Counter(
                    result["route_reason"]
                    for result in results
                    if result["route_reason"] is not None
                ).items()
            )
        ),
        "mean_retriever_attempts": round(
            sum(len(result["attempted_retrievers"]) for result in results) / count,
            4,
        )
        if count
        else 0.0,
        "evaluation_count": count,
    }


def _case_summary(case: RetrievalCase, results: list[dict[str, Any]]) -> dict[str, Any]:
    """只聚合指定 case 的重复结果，生成 HTML 表格数据。"""
    selected = [result for result in results if result["case_id"] == case.case_id]
    return {
        "case_id": case.case_id,
        "category": case.category,
        "query": case.query,
        "relevant_paths": list(case.relevant_paths),
        "mean_recall_at_1": _mean(selected, "recall_at_1"),
        "mean_recall_at_5": _mean(selected, "recall_at_5"),
        "mean_precision_at_5": _mean(selected, "precision_at_5"),
        "mean_reciprocal_rank": _mean(selected, "reciprocal_rank"),
        "latency_p50_ms": _percentile(
            [float(result["latency_ms"]) for result in selected], 50
        ),
        "latency_p95_ms": _percentile(
            [float(result["latency_ms"]) for result in selected], 95
        ),
    }


def _mean(results: list[dict[str, Any]], field: str) -> float:
    """求数字字段均值并保留四位小数，空输入返回 0。"""
    if not results:
        return 0.0
    return round(sum(float(result[field]) for result in results) / len(results), 4)


def _percentile(values: list[float], percent: int) -> float:
    """用 nearest-rank 方法计算百分位并保留三位小数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(percent / 100 * len(ordered)) - 1)
    return round(ordered[index], 3)


def _ensure_new_run_directory(output_dir: Path) -> None:
    """拒绝覆盖 workspace 或既有 JSON/HTML 报告，保护评测可追溯性。"""
    reserved = (
        output_dir / "workspace",
        output_dir / "retrieval-report.json",
        output_dir / "retrieval-report.html",
    )
    if any(path.exists() for path in reserved):
        raise FileExistsError(
            f"Retrieval output already contains run artifacts: {output_dir}"
        )
