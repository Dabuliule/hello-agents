from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import codecraft.tool.builtin.patch as patch_module
from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox.policy import SandboxMode
from codecraft.schema.tool import ToolCall
from codecraft.tool.base import ToolContext
from codecraft.tool.builtin.patch import ApplyPatchTool


def _tool_context(root: Path, patch: str) -> ToolContext:
    turn_context = TurnContext(
        session_id="ses_atomic_patch",
        turn_id="turn_atomic_patch",
        cwd=root,
        model="mock-model",
        model_provider="mock",
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        network_access=False,
        available_tools=[],
        max_tool_calls=10,
        max_tool_output_chars=80_000,
        created_at=datetime.now(UTC),
    )
    return ToolContext(
        context=turn_context,
        call=ToolCall(
            call_id="call_atomic_patch",
            name="apply_patch",
            arguments={"patch": patch},
        ),
    )


def _run_patch(root: Path, patch: str):
    tool = ApplyPatchTool()
    context = _tool_context(root, patch)
    arguments = tool.args_schema.model_validate(context.call.arguments)
    return asyncio.run(tool.arun(arguments, context))


def test_apply_patch_validates_every_target_before_writing(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")
    patch = """--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-alpha
+updated alpha
--- a/second.txt
+++ b/second.txt
@@ -1 +1 @@
-not beta
+updated beta
"""

    result = _run_patch(tmp_path, patch)

    assert result.success is False
    assert result.error == "patch_conflict"
    assert first.read_text(encoding="utf-8") == "alpha\n"
    assert second.read_text(encoding="utf-8") == "beta\n"


def test_apply_patch_rolls_back_when_a_later_atomic_write_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")
    patch = """--- a/first.txt
+++ b/first.txt
@@ -1 +1 @@
-alpha
+updated alpha
--- a/second.txt
+++ b/second.txt
@@ -1 +1 @@
-beta
+updated beta
"""
    real_atomic_write = patch_module.atomic_write_text
    call_count = 0

    def fail_second_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated write failure")
        real_atomic_write(path, content, encoding=encoding)

    monkeypatch.setattr(patch_module, "atomic_write_text", fail_second_write)

    result = _run_patch(tmp_path, patch)

    assert result.success is False
    assert result.error == "patch_write_failed"
    assert "rollback_errors" not in result.metadata
    assert first.read_text(encoding="utf-8") == "alpha\n"
    assert second.read_text(encoding="utf-8") == "beta\n"
