from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.ids import new_id
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.core.trace_report import build_trace_report, render_trace_json
from codecraft.eval.metrics import (
    aggregate_metrics,
    classify_failure,
    summarize_events,
    summarize_tasks,
)
from codecraft.eval.suite import (
    EVAL_SUITE_NAME,
    EvalTask,
    evaluate_task,
    seed_workspace,
)
from codecraft.llm import LLMProviderRegistry
from codecraft.sandbox import SandboxMode
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import EvalSessionContext, SessionConfig, SessionSource
from codecraft.tool import (
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    ToolRegistry,
    WorkspaceSearchTool,
    WriteFileTool,
)

EVAL_REPORT_SCHEMA_VERSION = 2


async def run_eval_suite(
    *,
    tasks: Sequence[EvalTask],
    base_config: SessionConfig,
    llm_providers: LLMProviderRegistry,
    output_dir: Path,
    repeat: int = 1,
    on_task_complete: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """顺序运行固定任务的全部 attempts，并返回可复现聚合报告。

    Args:
        tasks: task_id 唯一的评测任务。
        base_config: 提供模型、预算等基础设置；安全/路径字段会按 attempt 覆盖。
        llm_providers: 真实或测试 Provider Registry。
        output_dir: 不得含已有 run artifacts 的输出根。
        repeat: 每个 Task 重复次数。
        on_task_complete: 每个 attempt 完成后的同步进度回调。

    Returns:
        含 run、metrics、task summaries 和逐 attempt results 的 schema v2 报告。

    Raises:
        ValueError: repeat 非正或 task_id 重复。
        FileExistsError: 输出根含上次评测保留目录/报告。

    Tasks 严格顺序运行，避免模型并发、磁盘压力和 Provider 限流使延迟指标失去
    可比性；重复 attempt 仍各有独立 workspace、Session ID 和 Trace。这里的可复现
    指 fixture、权限、工具、grader 和调度条件稳定，不保证远端 LLM 输出确定；repeat
    用于观察波动，当前报告不计算 pass@k 或置信区间。
    """
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("eval tasks must have unique task ids")

    run_id = new_id("eval_")
    started_at = datetime.now(UTC)
    started = monotonic()
    output_dir = output_dir.expanduser().resolve()
    _ensure_new_run_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workspaces").mkdir(exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)

    schedule = [(task, attempt) for task in tasks for attempt in range(1, repeat + 1)]
    results: list[dict[str, Any]] = []
    for index, (task, attempt) in enumerate(schedule, start=1):
        result = await _run_task(
            task=task,
            attempt=attempt,
            base_config=base_config,
            llm_providers=llm_providers,
            output_dir=output_dir,
            run_id=run_id,
        )
        results.append(result)
        if on_task_complete is not None:
            on_task_complete(index, len(schedule), result)

    finished_at = datetime.now(UTC)
    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "suite": EVAL_SUITE_NAME,
            "model_provider": base_config.model_provider,
            "model": base_config.model,
            "repeat": repeat,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": int((monotonic() - started) * 1000),
            "output_dir": str(output_dir),
        },
        "metrics": aggregate_metrics(results, task_count=len(tasks), repeat=repeat),
        "tasks": summarize_tasks(results),
        "results": results,
    }


async def _run_task(
    *,
    task: EvalTask,
    attempt: int,
    base_config: SessionConfig,
    llm_providers: LLMProviderRegistry,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    """在隔离 workspace 运行一次真实 AgentRuntime，并以文件状态判分。

    Session 强制 source=cli_eval、approval=never、workspace_write、network=False，
    移除 user_instructions，避免本机交互和个人 Prompt 污染基准。只注册文件、
    patch 与检索工具，不提供 Bash，降低环境差异。

    Runtime 异常被记录但 finally 始终 close；Turn 终态后重新从 Store 读取事件，
    确保指标和 Trace 基于持久化事实而非暂存队列。模型 answer 仅进报告供诊断，
    correctness 由 Turn 终态与 workspace checks 共同决定。
    """
    workspace = output_dir / "workspaces" / task.task_id / f"attempt-{attempt:02d}"
    seed_workspace(task, workspace)
    session_id = new_id("ses_eval_")
    config = SessionConfig.model_validate(
        {
            **base_config.model_dump(mode="python"),
            "session_id": session_id,
            "source": SessionSource.CLI_EVAL,
            "cwd": workspace,
            "codecraft_home": output_dir / ".codecraft",
            "approval_policy": ApprovalPolicy.NEVER,
            "sandbox_mode": SandboxMode.WORKSPACE_WRITE,
            "network_access": False,
            "user_instructions": None,
            "created_at": datetime.now(UTC),
            "evaluation": EvalSessionContext(
                run_id=run_id,
                task_id=task.task_id,
                attempt=attempt,
            ),
        }
    )
    runtime = AgentRuntime(
        session_store=SessionStore(config.codecraft_home),
        llm_providers=llm_providers,
        tool_registry=_eval_tool_registry(),
        approval_manager=ApprovalManager(),
    )
    started = monotonic()
    events: list[RuntimeEvent] = []
    runtime_error: str | None = None

    try:
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message(new_id("inp_eval_"), task.prompt))
        while True:
            event = await thread.next_event()
            events.append(event)
            if event.type in {
                RuntimeEventType.TURN_FINISHED,
                RuntimeEventType.TURN_ABORTED,
            }:
                await thread.wait_until_idle()
                break
        events = (await thread.read_snapshot()).events
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        await runtime.close()

    checks = evaluate_task(task, workspace)
    final_status = _final_status(events)
    event_metrics = summarize_events(events)
    failure_type = classify_failure(
        events=events,
        runtime_error=runtime_error,
        final_status=final_status,
        checks=checks,
        tool_failure_count=event_metrics["tool_failure_count"],
    )
    passed = failure_type is None
    trace_report = build_trace_report(session_id, events)
    trace_path = (
        output_dir / "traces" / f"{task.task_id}.attempt-{attempt:02d}.trace.json"
    )
    trace_path.write_text(render_trace_json(trace_report), encoding="utf-8")

    return {
        "task_id": task.task_id,
        "attempt": attempt,
        "title": task.title,
        "category": task.category,
        "prompt": task.prompt,
        "status": "passed" if passed else "failed",
        "duration_ms": int((monotonic() - started) * 1000),
        "session_id": session_id,
        "final_status": final_status,
        "answer": _final_answer(events),
        "tool_call_count": event_metrics["tool_call_count"],
        "tool_failure_count": event_metrics["tool_failure_count"],
        "error_count": event_metrics["error_count"],
        "token_usage": event_metrics["token_usage"],
        "failure_type": failure_type,
        "checks": checks,
        "workspace": str(workspace),
        "trace_json": str(trace_path.relative_to(output_dir)),
        "error": runtime_error,
    }


def _eval_tool_registry() -> ToolRegistry:
    """构造不含 Bash/网络/MCP 的确定性读写、patch 和搜索工具集。"""
    return ToolRegistry(
        [
            ReadFileTool(),
            ListFilesTool(),
            WorkspaceSearchTool(),
            WriteFileTool(),
            ApplyPatchTool(),
        ]
    )


def _ensure_new_run_directory(output_dir: Path) -> None:
    """拒绝覆盖 .codecraft、workspaces、traces 或既有报告。"""
    reserved = (
        output_dir / ".codecraft",
        output_dir / "workspaces",
        output_dir / "traces",
        output_dir / "eval-report.json",
        output_dir / "eval-report.html",
    )
    if any(path.exists() for path in reserved):
        raise FileExistsError(
            f"Evaluation output already contains run artifacts: {output_dir}"
        )


def _final_status(events: list[RuntimeEvent]) -> str:
    """反向推导最近 finish/abort/error；完全无终态按 error。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_FINISHED:
            return str(event.payload.get("status") or "success")
        if event.type == RuntimeEventType.TURN_ABORTED:
            return "aborted"
        if event.type == RuntimeEventType.ERROR:
            return "error"
    return "error"


def _final_answer(events: list[RuntimeEvent]) -> str:
    """读取最近 TURN_FINISHED answer；未成功完成返回空。"""
    for event in reversed(events):
        if event.type == RuntimeEventType.TURN_FINISHED:
            return str(event.payload.get("answer") or "")
    return ""
