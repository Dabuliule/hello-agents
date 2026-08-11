from __future__ import annotations

from collections import Counter
from math import ceil
from typing import Any

from codecraft.schema.event import RuntimeEvent, RuntimeEventType

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "total_tokens",
)


def summarize_events(events: list[RuntimeEvent]) -> dict[str, Any]:
    """从一次 Session 事件聚合 Tool 终态、错误和标准 Token 字段。

    total_tokens 缺失时只用 input+output 回退，不把 reasoning/cached 再重复相加；
    布尔、负数和非数字安全归零。
    """
    tool_results = [
        event for event in events if event.type == RuntimeEventType.TOOL_CALL_FINISHED
    ]
    tool_failures = [
        event
        for event in tool_results
        if isinstance(event.payload.get("result"), dict)
        and event.payload["result"].get("success") is not True
    ]
    token_usage = {field: 0 for field in TOKEN_FIELDS}
    for event in events:
        if event.type != RuntimeEventType.TOKEN_COUNT:
            continue
        for field in TOKEN_FIELDS[:-1]:
            token_usage[field] += _non_negative_int(event.payload.get(field))
        total = event.payload.get("total_tokens")
        if _is_number(total):
            token_usage["total_tokens"] += _non_negative_int(total)
        else:
            token_usage["total_tokens"] += sum(
                _non_negative_int(event.payload.get(field))
                for field in ("input_tokens", "output_tokens")
            )

    return {
        "tool_call_count": len(tool_results),
        "tool_failure_count": len(tool_failures),
        "error_count": sum(
            event.type in {RuntimeEventType.ERROR, RuntimeEventType.TURN_ABORTED}
            for event in events
        ),
        "token_usage": token_usage,
    }


def classify_failure(
    *,
    events: list[RuntimeEvent],
    runtime_error: str | None,
    final_status: str,
    checks: list[dict[str, Any]],
    tool_failure_count: int,
) -> str | None:
    """按 runtime→model→abort→tool→grader 优先级归类唯一失败类型。

    只有 runtime 无异常、Turn success 且全部 checks 通过才返回 None。分类顺序
    保证一次 attempt 不会同时计入多个 failure bucket。
    """
    if (
        runtime_error is None
        and final_status == "success"
        and all(check["passed"] for check in checks)
    ):
        return None
    if runtime_error is not None:
        return "runtime_error"

    abort_reason = _abort_reason(events)
    if abort_reason == "model_error" or (
        final_status == "error"
        and any(event.type == RuntimeEventType.ERROR for event in events)
    ):
        return "model_error"
    if final_status != "success":
        return "turn_aborted"
    if tool_failure_count:
        return "tool_failure"
    return "grader_failure"


def aggregate_metrics(
    results: list[dict[str, Any]],
    *,
    task_count: int,
    repeat: int,
) -> dict[str, Any]:
    """汇总成功率、nearest-rank 延迟、工具/错误、Token 与失败分布。"""
    passed_count = sum(result["status"] == "passed" for result in results)
    evaluation_count = len(results)
    token_usage = {
        field: sum(result["token_usage"][field] for result in results)
        for field in TOKEN_FIELDS
    }
    failures = Counter(
        result["failure_type"]
        for result in results
        if result["failure_type"] is not None
    )
    durations = [result["duration_ms"] for result in results]
    return {
        "task_count": task_count,
        "repeat": repeat,
        "evaluation_count": evaluation_count,
        "passed_count": passed_count,
        "failed_count": evaluation_count - passed_count,
        "success_rate": (
            passed_count / evaluation_count * 100 if evaluation_count else 0.0
        ),
        "duration_p50_ms": percentile(durations, 50),
        "duration_p95_ms": percentile(durations, 95),
        "tool_call_count": sum(result["tool_call_count"] for result in results),
        "tool_failure_count": sum(result["tool_failure_count"] for result in results),
        "error_count": sum(result["error_count"] for result in results),
        "token_usage": token_usage,
        "failure_counts": dict(sorted(failures.items())),
    }


def summarize_tasks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按首次任务顺序聚合重复 attempts 的成功率、延迟和总 Token。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["task_id"], []).append(result)

    summaries = []
    for task_results in grouped.values():
        first = task_results[0]
        passed_count = sum(result["status"] == "passed" for result in task_results)
        attempt_count = len(task_results)
        summaries.append(
            {
                "task_id": first["task_id"],
                "title": first["title"],
                "category": first["category"],
                "attempt_count": attempt_count,
                "passed_count": passed_count,
                "success_rate": passed_count / attempt_count * 100,
                "duration_p50_ms": percentile(
                    [result["duration_ms"] for result in task_results], 50
                ),
                "duration_p95_ms": percentile(
                    [result["duration_ms"] for result in task_results], 95
                ),
                "total_tokens": sum(
                    result["token_usage"]["total_tokens"] for result in task_results
                ),
            }
        )
    return summaries


def percentile(values: list[int], percent: int) -> int:
    """用 nearest-rank 返回整数百分位；空输入返回 0。"""
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def _abort_reason(events: list[RuntimeEvent]) -> str | None:
    """反向读取最近 TURN_ABORTED 的 reason。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_ABORTED:
            reason = event.payload.get("reason")
            return str(reason) if reason else None
    return None


def _non_negative_int(value: Any) -> int:
    """把非布尔数字转成不小于零的 int，其余返回 0。"""
    if not _is_number(value):
        return 0
    return max(0, int(value))


def _is_number(value: Any) -> bool:
    """判断 int/float 且显式排除 Python 的 bool-int 子类关系。"""
    return not isinstance(value, bool) and isinstance(value, (int, float))
