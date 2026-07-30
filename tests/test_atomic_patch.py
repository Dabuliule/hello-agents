from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import codecraft.tool.builtin.filesystem as filesystem_module
import codecraft.tool.builtin.patch as patch_module
from codecraft.approval.policy import ApprovalPolicy
from codecraft.core.turn_context import TurnContext
from codecraft.sandbox.policy import SandboxMode
from codecraft.schema.tool import ToolCall
from codecraft.tool.base import ToolContext
from codecraft.tool.builtin.filesystem import ListFilesTool, WriteFileTool
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


def _run_write(root: Path, *, content: str):
    tool = WriteFileTool()
    turn_context = _tool_context(root, "").context
    call = ToolCall(
        call_id="call_atomic_write",
        name="write_file",
        arguments={"path": "target.txt", "content": content},
    )
    context = ToolContext(context=turn_context, call=call)
    arguments = tool.args_schema.model_validate(call.arguments)
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


def test_write_file_preserves_existing_content_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")

    def fail_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        del path, content, encoding
        raise OSError("simulated replace failure")

    monkeypatch.setattr(filesystem_module, "atomic_write_text", fail_write)

    result = _run_write(tmp_path, content="after\n")

    assert result.success is False
    assert result.error == "file_write_error"
    assert target.read_text(encoding="utf-8") == "before\n"


def test_write_file_skips_disk_write_when_content_is_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("same\n", encoding="utf-8")

    def unexpected_write(path: Path, content: str, *, encoding: str = "utf-8") -> None:
        del path, content, encoding
        raise AssertionError("unchanged content must not be rewritten")

    monkeypatch.setattr(filesystem_module, "atomic_write_text", unexpected_write)

    result = _run_write(tmp_path, content="same\n")

    assert result.success is True
    assert result.data["status"] == "unchanged"
    assert result.data["changed"] is False
    assert target.read_text(encoding="utf-8") == "same\n"


def test_list_files_orders_each_directory_before_bounded_traversal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "z-last.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a-first.txt").write_text("a", encoding="utf-8")
    real_scandir = filesystem_module.os.scandir
    with real_scandir(tmp_path) as scanner:
        reversed_entries = sorted(scanner, key=lambda item: item.name, reverse=True)

    class ReversedScandir:
        def __enter__(self):
            return iter(reversed_entries)

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(
        filesystem_module.os,
        "scandir",
        lambda path: ReversedScandir(),
    )

    entries = list(ListFilesTool._iter_entries(tmp_path, recursive=False))

    assert [entry.name for entry in entries] == ["a-first.txt", "z-last.txt"]
