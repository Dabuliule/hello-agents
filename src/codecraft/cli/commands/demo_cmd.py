from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from codecraft.approval.policy import ApprovalPolicy
from codecraft.cli import bootstrap
from codecraft.cli.commands.common import build_event_renderer, render_startup_error
from codecraft.cli.options import CodecraftHomeOption
from codecraft.cli.runtime_runner import submit_user_message
from codecraft.core.errors import CodecraftError
from codecraft.core.ids import new_id
from codecraft.llm import LLMProviderRegistry, MockProvider, ModelEvent, ModelEventType
from codecraft.prompt import BASE_INSTRUCTIONS
from codecraft.sandbox import SandboxBackendType, SandboxMode
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSource


_DEMO_FILE = "greeting.py"
_DEMO_SOURCE = 'def greeting(name: str) -> str:\n    return f"hello, {name}"\n'
_DEMO_RESULT_SOURCE = 'def greeting(name: str) -> str:\n    return f"Hello, {name}!"\n'
_DEMO_PATCH = """\
--- a/greeting.py
+++ b/greeting.py
@@ -1,2 +1,2 @@
 def greeting(name: str) -> str:
-    return f"hello, {name}"
+    return f"Hello, {name}!"
"""


def register_demo_command(app: typer.Typer) -> None:
    @app.command("demo")
    def demo_command(
        codecraft_home: CodecraftHomeOption = Path("~/.codecraft"),
        debug: Annotated[
            bool,
            typer.Option("--debug", help="Show verbose runtime events."),
        ] = False,
    ) -> None:
        """Run a deterministic governed-editing demo without an API key."""
        exit_code = asyncio.run(run_demo(codecraft_home=codecraft_home, debug=debug))
        if exit_code:
            raise typer.Exit(code=exit_code)


async def run_demo(*, codecraft_home: Path, debug: bool = False) -> int:
    session_id = new_id("ses_demo_")
    home = codecraft_home.expanduser().resolve()
    workspace = home / "demos" / session_id
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / _DEMO_FILE).write_text(_DEMO_SOURCE, encoding="utf-8")
    config = _demo_config(
        session_id=session_id,
        workspace=workspace,
        codecraft_home=home,
    )
    runtime = bootstrap.build_runtime(
        config,
        llm_providers=LLMProviderRegistry([MockProvider(_demo_script())]),
    )
    renderer = build_event_renderer(debug=debug)
    renderer.console.print(
        "CodeCraft demo: inspect a file, request approval, apply a patch, and persist the trace."
    )
    renderer.console.print(f"workspace: {workspace}", style="muted", markup=False)
    try:
        thread = await runtime.create_thread(config)
        exit_code = await submit_user_message(
            thread,
            renderer,
            "Improve the greeting while preserving the function signature.",
        )
        snapshot = await thread.read_snapshot()
        final_source = await asyncio.to_thread(
            (workspace / _DEMO_FILE).read_text,
            encoding="utf-8",
        )
        edit_applied = final_source == _DEMO_RESULT_SOURCE
        trace_verified = _demo_trace_succeeded(snapshot.events)
    except CodecraftError as exc:
        render_startup_error(exc)
        return 1
    finally:
        await runtime.close()

    renderer.console.print(f"session: {session_id}", style="muted", markup=False)
    renderer.console.print(
        f"inspect: codecraft inspect {session_id} --events",
        style="muted",
        markup=False,
    )
    if exit_code == 0 and edit_applied and trace_verified:
        renderer.console.print("demo result: governed edit applied", style="success")
        return 0

    if edit_applied:
        renderer.console.print(
            "demo result: edit applied, but governed trace verification failed",
            style="error",
        )
        return exit_code or 3

    renderer.console.print(
        "demo result: no edit was applied; inspect the approval and tool outcomes",
        style="warning",
    )
    return exit_code or 2


def _demo_config(
    *,
    session_id: str,
    workspace: Path,
    codecraft_home: Path,
) -> SessionConfig:
    return SessionConfig(
        session_id=session_id,
        source=SessionSource.CLI_DEMO,
        cwd=workspace,
        codecraft_home=codecraft_home,
        model="deterministic-demo",
        model_provider="mock",
        approval_policy=ApprovalPolicy.ON_REQUEST,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        sandbox_backend=SandboxBackendType.PROCESS,
        base_instructions=BASE_INSTRUCTIONS,
    )


def _demo_script() -> list[ModelEvent]:
    return [
        ModelEvent(
            type=ModelEventType.MESSAGE_DELTA,
            payload={"text": "I will inspect the target before editing it."},
        ),
        ModelEvent(
            type=ModelEventType.TOOL_CALL,
            payload={
                "call_id": "call_demo_read",
                "name": "read_file",
                "arguments": {"path": _DEMO_FILE},
            },
        ),
        ModelEvent(type=ModelEventType.COMPLETED),
        ModelEvent(
            type=ModelEventType.MESSAGE_DELTA,
            payload={"text": "The change is small and can be applied as one patch."},
        ),
        ModelEvent(
            type=ModelEventType.TOOL_CALL,
            payload={
                "call_id": "call_demo_patch",
                "name": "apply_patch",
                "arguments": {"patch": _DEMO_PATCH},
            },
        ),
        ModelEvent(type=ModelEventType.COMPLETED),
        ModelEvent(
            type=ModelEventType.MESSAGE_COMPLETED,
            payload={
                "text": (
                    "The governed workflow has finished. The tool result, approval "
                    "decision, and final state are available in the session trace."
                )
            },
        ),
        ModelEvent(type=ModelEventType.COMPLETED),
    ]


def _demo_trace_succeeded(events: Sequence[RuntimeEvent]) -> bool:
    read_index = _successful_tool_result_index(events, "call_demo_read")
    patch_index = _successful_tool_result_index(events, "call_demo_patch")
    approval_indices = _approved_patch_indices(events)
    applied_index = _patch_applied_index(events)
    if (
        read_index is None
        or patch_index is None
        or approval_indices is None
        or applied_index is None
    ):
        return False
    requested_index, decided_index = approval_indices
    return read_index < requested_index < decided_index < patch_index < applied_index


def _successful_tool_result_index(
    events: Sequence[RuntimeEvent],
    call_id: str,
) -> int | None:
    for index, event in enumerate(events):
        if event.type != RuntimeEventType.TOOL_CALL_FINISHED:
            continue
        if event.payload.get("call_id") != call_id:
            continue
        result = event.payload.get("result")
        if isinstance(result, dict) and result.get("success") is True:
            return index
    return None


def _approved_patch_indices(
    events: Sequence[RuntimeEvent],
) -> tuple[int, int] | None:
    for requested_index, event in enumerate(events):
        if event.type != RuntimeEventType.APPROVAL_REQUESTED:
            continue
        if event.payload.get("call_id") != "call_demo_patch":
            continue
        approval_id = event.payload.get("approval_id")
        for decided_index, decision in enumerate(
            events[requested_index + 1 :],
            start=requested_index + 1,
        ):
            if decision.type != RuntimeEventType.APPROVAL_DECIDED:
                continue
            if decision.payload.get("approval_id") != approval_id:
                continue
            if decision.payload.get("approved") is True:
                return requested_index, decided_index
            return None
    return None


def _patch_applied_index(events: Sequence[RuntimeEvent]) -> int | None:
    for index, event in enumerate(events):
        if (
            event.type == RuntimeEventType.PATCH_APPLIED
            and event.payload.get("call_id") == "call_demo_patch"
        ):
            return index
    return None
