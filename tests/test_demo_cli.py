from __future__ import annotations

import json

from typer.testing import CliRunner

from codecraft.cli.app import app


runner = CliRunner()


def _load_events(codecraft_home):
    logs = list((codecraft_home / "sessions").glob("**/ses_demo_*.jsonl"))
    assert len(logs) == 1
    return [json.loads(line) for line in logs[0].read_text().splitlines()]


def _event_index(events, event_type, *, call_id=None):
    return next(
        index
        for index, event in enumerate(events)
        if event["type"] == event_type
        and (call_id is None or event["payload"].get("call_id") == call_id)
    )


def test_demo_runs_without_an_api_key_and_persists_the_governed_edit(tmp_path):
    codecraft_home = tmp_path / ".codecraft"

    result = runner.invoke(
        app,
        ["demo", "--codecraft-home", str(codecraft_home)],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "CodeCraft demo" in result.output
    assert "approval required" in result.output
    assert "approval approved" in result.output
    assert "demo result: governed edit applied" in result.output

    workspaces = list((codecraft_home / "demos").glob("ses_demo_*"))
    assert len(workspaces) == 1
    assert (workspaces[0] / "greeting.py").read_text(encoding="utf-8") == (
        'def greeting(name: str) -> str:\n    return f"Hello, {name}!"\n'
    )

    events = _load_events(codecraft_home)
    assert events[0]["payload"]["config"]["source"] == "cli_demo"
    read_index = _event_index(events, "tool_call_finished", call_id="call_demo_read")
    requested_index = _event_index(
        events, "approval_requested", call_id="call_demo_patch"
    )
    decided_index = _event_index(events, "approval_decided")
    patch_index = _event_index(events, "tool_call_finished", call_id="call_demo_patch")
    applied_index = _event_index(events, "patch_applied", call_id="call_demo_patch")
    assert read_index < requested_index < decided_index < patch_index < applied_index
    assert events[read_index]["payload"]["result"]["success"] is True
    assert events[decided_index]["payload"]["approved"] is True
    assert events[patch_index]["payload"]["result"]["success"] is True


def test_demo_reports_a_rejected_edit_without_claiming_success(tmp_path):
    codecraft_home = tmp_path / ".codecraft"

    result = runner.invoke(
        app,
        ["demo", "--codecraft-home", str(codecraft_home)],
        input="n\n",
    )

    assert result.exit_code == 2, result.output
    assert "approval rejected" in result.output
    assert "demo result: no edit was applied" in result.output
    workspace = next((codecraft_home / "demos").glob("ses_demo_*"))
    assert (workspace / "greeting.py").read_text(encoding="utf-8") == (
        'def greeting(name: str) -> str:\n    return f"hello, {name}"\n'
    )

    events = _load_events(codecraft_home)
    decided_index = _event_index(events, "approval_decided")
    patch_index = _event_index(events, "tool_call_finished", call_id="call_demo_patch")
    assert events[decided_index]["payload"]["approved"] is False
    assert events[patch_index]["payload"]["result"]["success"] is False
    assert not any(event["type"] == "patch_applied" for event in events)
