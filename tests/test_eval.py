from __future__ import annotations

import json

from typer.testing import CliRunner

from codecraft.cli import app as cli_app
from codecraft.cli.app import app
from codecraft.eval.metrics import classify_failure, percentile, summarize_events
from codecraft.eval.suite import evaluate_task, get_eval_tasks, seed_workspace
from codecraft.llm import LLMProviderRegistry, MockProvider, ModelEvent, ModelEventType
from codecraft.schema.event import RuntimeEvent, RuntimeEventType


runner = CliRunner()


def test_builtin_eval_suite_has_ten_stable_tasks():
    tasks = get_eval_tasks()

    assert len(tasks) == 10
    assert len({task.task_id for task in tasks}) == 10
    assert {task.category for task in tasks} >= {
        "repository_search",
        "instruction_following",
        "structured_data",
        "multi_file_edit",
    }


def test_json_eval_checks_grade_structured_values(tmp_path):
    task = next(
        task for task in get_eval_tasks() if task.task_id == "update-json-settings"
    )
    seed_workspace(task, tmp_path)
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "service": "catalog",
                "features": {"search": True, "export": True},
                "retries": 3,
            }
        ),
        encoding="utf-8",
    )

    checks = evaluate_task(task, tmp_path)

    assert checks
    assert all(check["passed"] for check in checks)


def test_eval_metrics_derive_tokens_and_classify_failures():
    token_event = RuntimeEvent(
        event_id="evt_tokens",
        session_id="ses_eval",
        seq=1,
        type=RuntimeEventType.TOKEN_COUNT,
        payload={
            "input_tokens": 7,
            "output_tokens": 2,
            "reasoning_tokens": 1,
            "cached_input_tokens": 3,
        },
    )
    aborted_event = RuntimeEvent(
        event_id="evt_aborted",
        session_id="ses_eval",
        seq=2,
        type=RuntimeEventType.TURN_ABORTED,
        payload={
            "reason": "model_error",
            "message": "model error",
            "tool_calls": 0,
            "duration_ms": 1,
            "metadata": {},
        },
    )

    metrics = summarize_events([token_event])

    assert metrics["token_usage"] == {
        "input_tokens": 7,
        "output_tokens": 2,
        "reasoning_tokens": 1,
        "cached_input_tokens": 3,
        "total_tokens": 9,
    }
    assert percentile([40, 10, 30, 20], 50) == 20
    assert percentile([40, 10, 30, 20], 95) == 40
    assert (
        classify_failure(
            events=[aborted_event],
            runtime_error=None,
            final_status="aborted",
            checks=[{"passed": False}],
            tool_failure_count=0,
        )
        == "model_error"
    )
    assert (
        classify_failure(
            events=[],
            runtime_error=None,
            final_status="success",
            checks=[{"passed": False}],
            tool_failure_count=1,
        )
        == "tool_failure"
    )


def test_eval_list_prints_tasks_without_loading_provider():
    result = runner.invoke(app, ["eval", "--list"])

    assert result.exit_code == 0
    assert "create-welcome-file" in result.output
    assert "normalize-name-list" in result.output
    assert "repository_search" in result.output


def test_eval_command_runs_task_and_writes_reports(tmp_path, monkeypatch):
    provider = MockProvider(
        [
            ModelEvent(
                type=ModelEventType.TOKEN_COUNT,
                payload={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "reasoning_tokens": 1,
                    "cached_input_tokens": 1,
                    "total_tokens": 12,
                },
            ),
            ModelEvent(
                type=ModelEventType.TOOL_CALL,
                payload={
                    "call_id": "call_eval_write",
                    "name": "write_file",
                    "arguments": {
                        "path": "welcome.txt",
                        "content": "hello, codecraft\n",
                    },
                },
            ),
            ModelEvent(type=ModelEventType.COMPLETED),
            ModelEvent(
                type=ModelEventType.TOKEN_COUNT,
                payload={
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "reasoning_tokens": 0,
                    "cached_input_tokens": 2,
                    "total_tokens": 23,
                },
            ),
            ModelEvent(
                type=ModelEventType.MESSAGE_COMPLETED,
                payload={"text": "Created welcome.txt."},
            ),
            ModelEvent(type=ModelEventType.COMPLETED),
            ModelEvent(
                type=ModelEventType.TOKEN_COUNT,
                payload={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "reasoning_tokens": 1,
                    "cached_input_tokens": 1,
                    "total_tokens": 12,
                },
            ),
            ModelEvent(
                type=ModelEventType.TOOL_CALL,
                payload={
                    "call_id": "call_eval_write_2",
                    "name": "write_file",
                    "arguments": {
                        "path": "welcome.txt",
                        "content": "hello, codecraft\n",
                    },
                },
            ),
            ModelEvent(type=ModelEventType.COMPLETED),
            ModelEvent(
                type=ModelEventType.TOKEN_COUNT,
                payload={
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "reasoning_tokens": 0,
                    "cached_input_tokens": 2,
                    "total_tokens": 23,
                },
            ),
            ModelEvent(
                type=ModelEventType.MESSAGE_COMPLETED,
                payload={"text": "Created welcome.txt."},
            ),
            ModelEvent(type=ModelEventType.COMPLETED),
        ]
    )
    monkeypatch.setattr(
        cli_app,
        "_build_provider_registry",
        lambda _config: LLMProviderRegistry([provider]),
    )
    output_dir = tmp_path / "eval-run"

    result = runner.invoke(
        app,
        [
            "eval",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft-config"),
            "--task",
            "create-welcome-file",
            "--repeat",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "eval_success_rate: 100.0% (2/2)" in result.output
    json_path = output_dir / "eval-report.json"
    html_path = output_dir / "eval-report.html"
    assert json_path.is_file()
    assert html_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    metrics = report["metrics"]
    assert metrics["task_count"] == 1
    assert metrics["repeat"] == 2
    assert metrics["evaluation_count"] == 2
    assert metrics["passed_count"] == 2
    assert metrics["failed_count"] == 0
    assert metrics["success_rate"] == 100.0
    assert metrics["tool_call_count"] == 2
    assert metrics["tool_failure_count"] == 0
    assert metrics["error_count"] == 0
    assert metrics["token_usage"] == {
        "input_tokens": 60,
        "output_tokens": 10,
        "reasoning_tokens": 2,
        "cached_input_tokens": 6,
        "total_tokens": 70,
    }
    assert metrics["failure_counts"] == {}
    assert metrics["duration_p50_ms"] >= 0
    assert metrics["duration_p95_ms"] >= metrics["duration_p50_ms"]
    assert report["tasks"][0]["attempt_count"] == 2
    assert report["tasks"][0]["passed_count"] == 2
    assert report["tasks"][0]["success_rate"] == 100.0
    assert report["tasks"][0]["total_tokens"] == 70
    assert [task_result["attempt"] for task_result in report["results"]] == [1, 2]
    for task_result in report["results"]:
        assert task_result["task_id"] == "create-welcome-file"
        assert task_result["status"] == "passed"
        assert task_result["final_status"] == "success"
        assert task_result["failure_type"] is None
        assert task_result["token_usage"]["total_tokens"] == 35
        assert all(check["passed"] for check in task_result["checks"])
        assert (output_dir / task_result["trace_json"]).is_file()
    session_configs = [
        json.loads(path.read_text(encoding="utf-8").splitlines()[0])["payload"][
            "config"
        ]
        for path in (output_dir / ".codecraft" / "sessions").rglob("*.jsonl")
    ]
    assert {
        (config["evaluation"]["task_id"], config["evaluation"]["attempt"])
        for config in session_configs
    } == {("create-welcome-file", 1), ("create-welcome-file", 2)}
    assert all("metadata" not in config for config in session_configs)
    assert "CodeCraft Eval" in html_path.read_text(encoding="utf-8")

    repeated = runner.invoke(
        app,
        [
            "eval",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft-config"),
            "--task",
            "create-welcome-file",
            "--repeat",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert repeated.exit_code == 2
    assert "already contains run artifacts" in repeated.output
    assert "Traceback" not in repeated.output


def test_eval_unknown_task_prints_friendly_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "eval",
            "--task",
            "missing-task",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 2
    assert "Unknown eval task(s): missing-task" in result.output
    assert "Traceback" not in result.output


def test_eval_duplicate_task_prints_friendly_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "eval",
            "--task",
            "create-welcome-file",
            "--task",
            "create-welcome-file",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 2
    assert "Duplicate eval task(s): create-welcome-file" in result.output
    assert "Traceback" not in result.output


def test_eval_command_reports_failed_deterministic_checks(tmp_path, monkeypatch):
    provider = MockProvider(
        [
            ModelEvent(
                type=ModelEventType.MESSAGE_COMPLETED,
                payload={"text": "Done without editing files."},
            ),
            ModelEvent(type=ModelEventType.COMPLETED),
        ]
    )
    monkeypatch.setattr(
        cli_app,
        "_build_provider_registry",
        lambda _config: LLMProviderRegistry([provider]),
    )
    output_dir = tmp_path / "failed-eval"

    result = runner.invoke(
        app,
        [
            "eval",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
            "--task",
            "create-welcome-file",
            "--output-dir",
            str(output_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert "eval_success_rate: 0.0% (0/1)" in result.output
    report = json.loads((output_dir / "eval-report.json").read_text(encoding="utf-8"))
    task_result = report["results"][0]
    assert task_result["status"] == "failed"
    assert task_result["final_status"] == "success"
    assert task_result["failure_type"] == "grader_failure"
    assert any(not check["passed"] for check in task_result["checks"])
    assert task_result["error"] is None
