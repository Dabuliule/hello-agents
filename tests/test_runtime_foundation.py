from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from codecraft.approval.manager import (
    ApprovalManager,
    ApprovalRequest,
    AutoApprovalReviewer,
    DenyApprovalReviewer,
)
from codecraft.approval.policy import ApprovalPolicy
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.core.errors import (
    SessionError,
    SessionRestoreError,
    WorkspaceAccessError,
)
from codecraft.core.event_bus import EventBus
from codecraft.core.ids import new_id
from codecraft.core.runtime import AgentRuntime
from codecraft.core.session_store import SessionStore
from codecraft.core.turn_context import TurnContext
from codecraft.llm import (
    DeepSeekProvider,
    LLMConfigError,
    LLMProvider,
    LLMProviderRegistry,
    ModelEvent,
    ModelEventType,
    ModelMessage,
    ModelMessageType,
    ModelRequest,
    ModelRole,
    MockProvider,
    OpenAIProvider,
    QwenProvider,
)
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.input import SessionInput
from codecraft.schema.session import SessionConfig, SessionSnapshot, SessionSource
from codecraft.schema.tool import ToolCall, ToolEffect, ToolResult, ToolSpec
from codecraft.tool import (
    ApplyPatchTool,
    BashTool,
    BaseTool,
    ListFilesTool,
    ReadFileTool,
    ToolContext,
    ToolRegistry,
    WorkspaceSearchTool,
    WorkspaceGuard,
    WriteFileTool,
)
from codecraft.tool.runner import ToolRunner
from codecraft.sandbox import CommandPolicy, CommandRisk, SandboxMode, SandboxPolicy
from codecraft.prompt import BASE_INSTRUCTIONS, InstructionLoader


def make_config(tmp_path) -> SessionConfig:
    return SessionConfig(
        session_id="ses_test",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy="never",
        sandbox_mode="workspace_write",
    )


def test_base_instructions_are_loaded_from_markdown_resource():
    assert BASE_INSTRUCTIONS.startswith("# CodeCraft Base Instructions")
    assert "Respect the active sandbox and approval policy." in BASE_INSTRUCTIONS


async def next_event_of_type(thread, event_type: RuntimeEventType) -> RuntimeEvent:
    while True:
        event = await thread.next_event()
        if event.type == event_type:
            return event


def test_runtime_event_is_json_serializable(tmp_path):
    config = make_config(tmp_path)
    event = RuntimeEvent(
        event_id="evt_test",
        session_id=config.session_id,
        seq=1,
        type=RuntimeEventType.SESSION_STARTED,
        payload={"config": config.model_dump(mode="json")},
    )

    encoded = event.model_dump_json()
    decoded = RuntimeEvent.model_validate_json(encoded)

    assert decoded.type == RuntimeEventType.SESSION_STARTED
    assert decoded.schema_version == 1
    assert decoded.seq == 1
    assert decoded.payload["config"]["session_id"] == "ses_test"


def test_runtime_event_sanitizes_invalid_unicode_payload():
    event = RuntimeEvent(
        event_id="evt_test",
        session_id="ses_test",
        seq=1,
        type=RuntimeEventType.MODEL_TOOL_CALL,
        payload={
            "call_id": "call_test",
            "name": "echo",
            "arguments": {
                "text": "bad\udce4text",
                "nested": {"\udce5": ["ok\udce6"]},
            },
        },
    )

    encoded = event.model_dump_json()
    decoded = RuntimeEvent.model_validate_json(encoded)

    arguments = decoded.payload["arguments"]
    assert "\udce4" not in arguments["text"]
    assert "bad?text" == arguments["text"]
    assert "?" in arguments["nested"]
    assert arguments["nested"]["?"] == ["ok?"]


def test_runtime_event_redacts_sensitive_fields_without_hiding_env_names():
    event = RuntimeEvent(
        event_id="evt_secret",
        session_id="ses_test",
        seq=1,
        type=RuntimeEventType.MODEL_TOOL_CALL,
        payload={
            "call_id": "call_secret",
            "name": "secret_tool",
            "arguments": {
                "api_key": "secret-value",
                "nested": {"access_token": "token-value"},
                "model_api_key_env": "DASHSCOPE_API_KEY",
            },
        },
    )

    assert event.payload["arguments"]["api_key"] == "[REDACTED]"
    assert event.payload["arguments"]["nested"]["access_token"] == "[REDACTED]"
    assert event.payload["arguments"]["model_api_key_env"] == "DASHSCOPE_API_KEY"


def test_runtime_event_rejects_payload_for_the_wrong_event_type():
    with pytest.raises(ValueError, match="input_id"):
        RuntimeEvent(
            event_id="evt_wrong_payload",
            session_id="ses_test",
            seq=1,
            type=RuntimeEventType.USER_MESSAGE,
            payload={"text": "missing input id"},
        )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RuntimeEvent(
            event_id="evt_extra_payload",
            session_id="ses_test",
            seq=1,
            type=RuntimeEventType.ASSISTANT_MESSAGE,
            payload={"text": "hello", "unexpected": True},
        )


def test_runtime_event_payload_is_frozen():
    event = RuntimeEvent(
        event_id="evt_frozen",
        session_id="ses_test",
        turn_id="turn_test",
        seq=1,
        type=RuntimeEventType.ASSISTANT_MESSAGE,
        payload={"text": "immutable"},
    )

    with pytest.raises(ValueError, match="frozen"):
        event.payload.text = "changed"


def test_runtime_event_rejects_non_object_payload():
    with pytest.raises(ValueError, match="JSON object"):
        RuntimeEvent(
            event_id="evt_bad_payload",
            session_id="ses_test",
            seq=1,
            type=RuntimeEventType.SESSION_CLOSED,
            payload=["not", "an", "object"],
        )


def test_runtime_event_payload_serializer_respects_filters():
    event = RuntimeEvent(
        event_id="evt_error",
        session_id="ses_test",
        seq=1,
        type=RuntimeEventType.ERROR,
        payload={
            "code": "runtime_error",
            "message": "failed",
            "metadata": {"detail": "test"},
            "suggestion": None,
        },
    )

    without_none = event.model_dump(mode="json", exclude_none=True)
    without_metadata = event.model_dump(
        mode="json",
        exclude={"payload": {"metadata"}},
    )

    assert "suggestion" not in without_none["payload"]
    assert "metadata" not in without_metadata["payload"]


def test_session_started_payload_preserves_data_keys_that_look_sensitive(tmp_path):
    config_data = make_config(tmp_path).model_dump(mode="json")
    config_data["mcp_servers"] = {"token": {"command": "mcp-server"}}

    event = RuntimeEvent(
        event_id="evt_config",
        session_id="ses_test",
        seq=1,
        type=RuntimeEventType.SESSION_STARTED,
        payload={"config": config_data},
    )

    assert "token" in event.payload["config"]["mcp_servers"]


def test_persisted_session_config_does_not_require_cwd_to_still_exist(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = make_config(workspace)
    serialized = RuntimeEvent(
        event_id="evt_started",
        session_id=config.session_id,
        seq=1,
        type=RuntimeEventType.SESSION_STARTED,
        payload={"config": config.model_dump(mode="json")},
    ).model_dump_json()
    workspace.rmdir()

    restored = RuntimeEvent.model_validate_json(serialized)

    assert restored.payload["config"]["cwd"] == str(workspace)
    with pytest.raises(ValueError, match="existing directory"):
        restored_config = SessionConfig.model_validate(restored.payload["config"])
        restored_config.ensure_runtime_ready()


def test_runtime_event_requires_positive_seq():
    with pytest.raises(ValueError):
        RuntimeEvent(
            event_id="evt_bad",
            session_id="ses_test",
            seq=0,
            type=RuntimeEventType.SESSION_STARTED,
        )


def test_session_input_sanitizes_invalid_unicode_user_message():
    input = SessionInput.user_message("inp_test", "你能联网吗\udce4")

    assert input.payload.text == "你能联网吗?"
    input.model_dump_json()


def test_session_input_rejects_blank_messages_and_unknown_payload_fields():
    with pytest.raises(ValueError, match="must not be blank"):
        SessionInput.user_message("inp_blank", "   ")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        SessionInput.model_validate(
            {
                "input_id": "inp_extra",
                "type": "user_message",
                "payload": {"text": "hello", "unexpected": True},
            }
        )


def test_model_tool_call_event_requires_normalized_identity():
    with pytest.raises(ValueError, match="call_id"):
        ModelEvent(
            type=ModelEventType.TOOL_CALL,
            payload={"name": "read_file", "arguments": {"path": "README.md"}},
        )


def test_session_emit_rolls_back_seq_when_append_fails(tmp_path):
    class FailingStore(SessionStore):
        def __init__(self, codecraft_home):
            super().__init__(codecraft_home)
            self.calls = 0

        async def append_event(self, event: RuntimeEvent) -> None:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("append failed")
            await super().append_event(event)

    async def run_test() -> None:
        config = make_config(tmp_path)
        store = FailingStore(config.codecraft_home)
        runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)

        with pytest.raises(RuntimeError, match="append failed"):
            await thread.session.emit(
                RuntimeEventType.USER_MESSAGE,
                {"input_id": "inp_failed", "text": "failed"},
            )

        event = await thread.session.emit(
            RuntimeEventType.USER_MESSAGE,
            {"input_id": "inp_ok", "text": "ok"},
        )
        loaded = await store.load_events(config.session_id)

        assert event.seq == 2
        assert [item.seq for item in loaded] == [1, 2]

    asyncio.run(run_test())


def test_session_turn_error_preserves_codecraft_error_metadata(tmp_path):
    class FailingStore(SessionStore):
        async def append_event(self, event: RuntimeEvent) -> None:
            if event.type == RuntimeEventType.USER_MESSAGE:
                raise SessionError(
                    "failed to append session event",
                    code="session_event_append_failed",
                    metadata={
                        "event_type": event.type.value,
                        "seq": event.seq,
                        "cause": "synthetic failure",
                    },
                )
            await super().append_event(event)

    async def run_test() -> None:
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=FailingStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )
        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "failed"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        error = snapshot.events[-2]
        aborted = snapshot.events[-1]
        assert error.type == RuntimeEventType.ERROR
        assert error.payload["code"] == "session_event_append_failed"
        assert error.payload["metadata"]["event_type"] == "user_message"
        assert error.payload["metadata"]["cause"] == "synthetic failure"
        assert aborted.payload["reason"] == "session_event_append_failed"

    asyncio.run(run_test())


def test_event_bus_dispatches_runtime_events_in_subscription_order():
    async def run_test() -> None:
        calls: list[tuple[str, int]] = []
        bus = EventBus()

        async def first(event: RuntimeEvent) -> None:
            calls.append(("first", event.seq))

        async def second(event: RuntimeEvent) -> None:
            calls.append(("second", event.seq))

        bus.subscribe(first)
        bus.subscribe(second)
        await bus.emit(
            RuntimeEvent(
                event_id="evt_test",
                session_id="ses_test",
                seq=1,
                type=RuntimeEventType.SESSION_CLOSED,
            )
        )

        assert calls == [("first", 1), ("second", 1)]

    asyncio.run(run_test())


def test_tool_result_enforces_error_shape():
    assert ToolResult(success=True, content="ok").error is None

    with pytest.raises(ValueError):
        ToolResult(success=True, content="ok", error="unexpected")

    with pytest.raises(ValueError):
        ToolResult(success=False, content="failed")


def test_tool_registry_registers_and_lists_specs():
    class EchoArgs(BaseModel):
        text: str

    class EchoTool(BaseTool):
        name = "echo"
        description = "Echo text."
        args_schema = EchoArgs
        effects = {ToolEffect.READ_ONLY}

        async def arun(self, args: EchoArgs, context: ToolContext) -> ToolResult:
            return ToolResult(success=True, content=args.text)

    registry = ToolRegistry([EchoTool()])

    assert registry.get("echo").name == "echo"
    assert registry.specs() == [
        ToolSpec(
            name="echo",
            description="Echo text.",
            input_schema=EchoArgs.model_json_schema(),
            effects={ToolEffect.READ_ONLY},
        )
    ]


def test_command_policy_classifies_safe_prompt_and_deny_commands():
    policy = CommandPolicy()

    assert policy.classify("pwd").risk == CommandRisk.SAFE
    assert policy.classify("git status").risk == CommandRisk.SAFE
    assert policy.classify("python --version").risk == CommandRisk.SAFE
    assert policy.classify("rm temp.txt").risk == CommandRisk.PROMPT
    assert policy.classify("curl https://example.com").risk == CommandRisk.DENY
    assert policy.classify("git fetch").risk == CommandRisk.DENY
    assert (
        policy.classify("curl https://example.com", network_access=True).risk
        == CommandRisk.PROMPT
    )
    assert policy.classify("git fetch", network_access=True).risk == CommandRisk.PROMPT
    assert policy.classify("sudo true").risk == CommandRisk.DENY


@pytest.mark.parametrize(
    "command",
    [
        "pwd",
        "pwd -P",
        "ls",
        "ls -la",
        "rg --files",
        "rg 'workspace root'",
        "rg -n 'workspace root'",
        "rg 'left|right'",
        "python --version",
        "git status",
        "git branch --show-current",
        "git stash list",
        "git tag --list",
        "git remote -v",
    ],
)
def test_command_policy_allows_only_exact_read_only_shapes(command):
    assert CommandPolicy().classify(command).risk == CommandRisk.SAFE


@pytest.mark.parametrize(
    "command",
    [
        "cat README.md",
        "grep needle README.md",
        "sed 's/foo/bar/' README.md",
        "sed -n '5,10p' file.txt",
        "sed -i 's/foo/bar/' README.md",
        "pytest -q",
        "python -m pytest",
        "uv run pytest",
        "npm test",
        "mvn test",
        "git status --short",
        "git branch feature/unsafe",
        "git tag v1.0.0",
        "git remote add origin https://example.com/repo.git",
        "git -C /tmp status",
        "rg --pre cat needle",
        "rg needle src",
        "rg needle /etc",
        "ls src",
        "ls /etc",
    ],
)
def test_command_policy_requires_approval_for_non_exact_shapes(command):
    decision = CommandPolicy().classify(command)

    assert decision.risk == CommandRisk.PROMPT
    assert decision.requires_approval is True


@pytest.mark.parametrize(
    "command",
    [
        "pwd > output.txt",
        "pwd >> output.txt",
        "cat < README.md",
        "pwd &",
        "pwd\nls",
        "pwd | ls",
        "pwd && ls",
        "pwd; ls",
        "ls *",
        "pwd ${HOME}",
    ],
)
def test_command_policy_shell_control_and_expansion_require_approval(command):
    decision = CommandPolicy().classify(command)

    assert decision.risk == CommandRisk.PROMPT
    assert decision.requires_approval is True


@pytest.mark.parametrize(
    "command",
    [
        "echo $(id)",
        "echo `id`",
        "cat <(pwd)",
        'echo "$(id)"',
    ],
)
def test_command_policy_denies_opaque_shell_substitution(command):
    decision = CommandPolicy().classify(command)

    assert decision.risk == CommandRisk.DENY
    assert decision.requires_approval is False


def test_command_policy_sed_always_requires_approval():
    """sed scripts are not parsed deeply enough to prove they cannot execute or edit."""
    policy = CommandPolicy()

    assert policy.classify("sed 's/foo/bar/' README.md").risk == CommandRisk.PROMPT
    assert policy.classify("sed -n '5,10p' file.txt").risk == CommandRisk.PROMPT
    assert policy.classify("sed -i 's/foo/bar/' README.md").risk == CommandRisk.PROMPT
    assert (
        policy.classify("sed --in-place 's/foo/bar/' README.md").risk
        == CommandRisk.PROMPT
    )
    assert policy.classify("sed -ie 's/foo/bar/' README.md").risk == CommandRisk.PROMPT
    assert policy.classify("sed -Ei 's/foo/bar/' README.md").risk == CommandRisk.PROMPT


def test_command_policy_detects_shell_metacharacters():
    """Control syntax prompts even when every segment is otherwise read-only."""
    policy = CommandPolicy()

    # Safe command chained with a dangerous one → DENY.
    result = policy.classify("pwd && sudo true")
    assert result.risk == CommandRisk.DENY
    assert result.reason == "sudo is denied"

    # Safe command chained with a prompt one → PROMPT.
    result = policy.classify("pwd; rm file.txt")
    assert result.risk == CommandRisk.PROMPT
    assert "shell control syntax" in result.reason

    # Pipes and chains require approval instead of inheriting SAFE.
    result = policy.classify("ls | grep foo")
    assert result.risk == CommandRisk.PROMPT

    result = policy.classify("pwd && ls")
    assert result.risk == CommandRisk.PROMPT

    # Newlines cannot conceal a denied second command.
    result = policy.classify("pwd\nsudo true")
    assert result.risk == CommandRisk.DENY


def test_command_policy_destructive_rm_patterns_are_denied():
    """rm -rf with dangerous paths is always DENY, even for broader patterns."""
    policy = CommandPolicy()

    assert policy.classify("rm -rf /").risk == CommandRisk.DENY
    assert policy.classify("rm -rf /*").risk == CommandRisk.DENY
    assert policy.classify("rm -rf ~").risk == CommandRisk.DENY
    assert policy.classify("rm -rf *").risk == CommandRisk.DENY
    assert policy.classify("rm -fr /").risk == CommandRisk.DENY
    assert policy.classify("rm -r -f /").risk == CommandRisk.DENY
    assert policy.classify("rm --recursive --force /").risk == CommandRisk.DENY
    assert policy.classify("/bin/rm -rf /").risk == CommandRisk.DENY
    assert policy.classify("rm -rf /tmp/..").risk == CommandRisk.DENY
    assert policy.classify("rm -rf ./build/..").risk == CommandRisk.DENY
    assert policy.classify("rm -rf ./*").risk == CommandRisk.DENY
    assert policy.classify("rm -rf ../*").risk == CommandRisk.DENY
    assert policy.classify("rm -rf / specific-file").risk == CommandRisk.DENY
    assert policy.classify("rm -rf specific-file /").risk == CommandRisk.DENY
    assert policy.classify('rm -rf "$HOME"').risk == CommandRisk.DENY
    assert policy.classify("rm -rf ${TARGET}").risk == CommandRisk.DENY

    # rm without -rf on a specific file is PROMPT, not DENY.
    assert policy.classify("rm file.txt").risk == CommandRisk.PROMPT
    assert policy.classify("rm -r dir/").risk == CommandRisk.PROMPT

    # rm with -rf on a non-broad path is still PROMPT (not auto-DENY).
    assert policy.classify("rm -rf build/").risk == CommandRisk.PROMPT


def test_command_policy_expanded_safe_git_subcommands():
    """Only exact read-only git argv shapes are SAFE."""
    policy = CommandPolicy()

    assert policy.classify("git branch").risk == CommandRisk.SAFE
    assert policy.classify("git stash list").risk == CommandRisk.SAFE
    assert policy.classify("git tag").risk == CommandRisk.SAFE
    assert policy.classify("git remote -v").risk == CommandRisk.SAFE

    assert policy.classify("git branch new-branch").risk == CommandRisk.PROMPT
    assert policy.classify("git stash push").risk == CommandRisk.PROMPT
    assert policy.classify("git tag v1").risk == CommandRisk.PROMPT
    assert policy.classify("git remote add origin local").risk == CommandRisk.PROMPT

    # Network and destructive git subcommands are not SAFE.
    assert policy.classify("git fetch").risk == CommandRisk.DENY
    assert policy.classify("git fetch", network_access=True).risk == CommandRisk.PROMPT
    assert policy.classify("git push").risk == CommandRisk.DENY
    assert policy.classify("git push", network_access=True).risk == CommandRisk.PROMPT
    assert policy.classify("git commit -m wip").risk == CommandRisk.PROMPT


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/sudo true",
        "/bin/dd if=/dev/zero of=image",
        "/sbin/mkfs.ext4 /dev/test",
        "/usr/bin/curl https://example.com",
        "/usr/bin/git fetch",
        "git -C /tmp fetch",
    ],
)
def test_command_policy_classifies_absolute_executable_paths(command):
    assert CommandPolicy().classify(command).risk == CommandRisk.DENY


def test_approval_manager_defaults_to_deny_reviewer():
    manager = ApprovalManager()

    assert isinstance(manager.reviewer, DenyApprovalReviewer)
    decision = asyncio.run(
        manager.request(
            ApprovalRequest(
                approval_id="appr_default_deny",
                session_id="ses_default_deny",
                turn_id="turn_default_deny",
                call_id="call_default_deny",
                tool_name="write_file",
                arguments={"path": "blocked.txt"},
                reason="write_file has side effects",
                risk="prompt",
            )
        )
    )

    assert decision.approved is False
    assert decision.reason == "no approval reviewer is configured"


def test_runtime_and_tool_runner_inherit_deny_reviewer_default(tmp_path):
    registry = ToolRegistry()
    runner = ToolRunner(registry)
    runtime = AgentRuntime(
        session_store=SessionStore(tmp_path / ".codecraft"),
        llm_providers=LLMProviderRegistry([MockProvider(script=[])]),
        tool_registry=registry,
    )

    assert isinstance(runner.approval_manager.reviewer, DenyApprovalReviewer)
    assert isinstance(runtime.approval_manager.reviewer, DenyApprovalReviewer)


def test_tool_runner_default_reviewer_rejects_approval_request(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(
            update={"approval_policy": ApprovalPolicy.ON_REQUEST}
        )
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_default_deny",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        events = [
            event
            async for event in ToolRunner(ToolRegistry([WriteFileTool()])).run(
                ToolCall(
                    call_id="call_default_deny",
                    name="write_file",
                    arguments={"path": "blocked.txt", "content": "blocked"},
                ),
                context,
            )
        ]

        assert not (tmp_path / "blocked.txt").exists()
        assert [event.type for event in events] == [
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.APPROVAL_REQUESTED,
            RuntimeEventType.APPROVAL_DECIDED,
            RuntimeEventType.TOOL_CALL_FINISHED,
        ]
        assert events[2].payload["approved"] is False
        assert events[2].payload["reason"] == "no approval reviewer is configured"
        assert events[3].payload["result"]["error"] == "approval_denied"

    asyncio.run(run_test())


def test_instruction_loader_uses_cwd_as_instruction_boundary(tmp_path):
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    package.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("root agents", encoding="utf-8")
    (package / "CODECRAFT.md").write_text("package codecraft", encoding="utf-8")

    loaded = InstructionLoader().load_project_instructions(
        cwd=package,
    )

    assert loaded is not None
    assert "# CODECRAFT.md" in loaded
    assert "package codecraft" in loaded
    assert "root agents" not in loaded
    assert "scope: ." in loaded


def test_instruction_loader_applies_target_scopes_and_skips_escaped_symlinks(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    package.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("root rule", encoding="utf-8")
    (package / "AGENTS.md").write_text("package rule", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("outside secret", encoding="utf-8")
    (workspace / "CODECRAFT.md").symlink_to(outside)

    loaded = InstructionLoader().load_project_instructions(
        cwd=workspace,
        target_paths=[Path("pkg/source.py"), outside],
    )

    assert loaded is not None
    assert "root rule" in loaded
    assert "package rule" in loaded
    assert "outside secret" not in loaded
    assert loaded.index("root rule") < loaded.index("package rule")


def test_instruction_loader_bounds_large_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 10_000, encoding="utf-8")

    loaded = InstructionLoader(max_chars=200).load_project_instructions(
        cwd=tmp_path,
    )

    assert loaded is not None
    assert len(loaded) <= 200
    assert loaded.startswith("[earlier project instructions omitted]")


def test_workspace_guard_rejects_path_escape(tmp_path):
    guard = WorkspaceGuard(tmp_path)

    with pytest.raises(Exception, match="outside workspace"):
        guard.resolve_read_path("../outside.txt")


def test_read_file_and_list_files_tools(tmp_path):
    async def run_test() -> None:
        nested = tmp_path / "pkg"
        nested.mkdir()
        target = nested / "note.txt"
        target.write_text("hello tools\n", encoding="utf-8")
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )

        read_tool = ReadFileTool()
        list_tool = ListFilesTool()
        read_call = ToolCall(
            call_id="call_read",
            name="read_file",
            arguments={"path": "pkg/note.txt"},
        )
        list_call = ToolCall(
            call_id="call_list",
            name="list_files",
            arguments={"path": "pkg"},
        )

        read_result = await read_tool.arun(
            read_tool.args_schema.model_validate(read_call.arguments),
            ToolContext(context=context, call=read_call),
        )
        list_result = await list_tool.arun(
            list_tool.args_schema.model_validate(list_call.arguments),
            ToolContext(context=context, call=list_call),
        )

        assert read_result.success is True
        assert read_result.content == "hello tools\n"
        assert read_result.data["line_count"] == 1
        assert list_result.success is True
        assert list_result.content == "note.txt"

        limited_events = [
            event
            async for event in ToolRunner(ToolRegistry([read_tool])).run(
                read_call,
                context.model_copy(update={"max_tool_output_chars": 5}),
            )
        ]
        limited_result = limited_events[-1].payload["result"]
        assert limited_result["content"] == "hello"
        assert limited_result["metadata"]["content_truncated"] is True
        assert limited_result["metadata"]["original_content_chars"] == 12
        assert (
            "output truncated from 12 characters"
            in ToolResult.model_validate(limited_result).model_content()
        )

    asyncio.run(run_test())


def test_read_file_reads_only_the_bounded_prefix(tmp_path, monkeypatch):
    async def run_test() -> None:
        target = tmp_path / "large.txt"
        target.write_text("0123456789", encoding="utf-8")
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_bounded_read",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        read_sizes: list[int] = []
        original_open = Path.open

        class TrackingStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.stream.close()

            def fileno(self):
                return self.stream.fileno()

            def read(self, size=-1):
                read_sizes.append(size)
                return self.stream.read(size)

        def tracking_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            if path == target and args and args[0] == "r":
                return TrackingStream(stream)
            return stream

        monkeypatch.setattr(Path, "open", tracking_open)
        tool = ReadFileTool()
        call = ToolCall(
            call_id="call_bounded_read",
            name=tool.name,
            arguments={"path": "large.txt", "max_chars": 5},
        )

        result = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )

        assert result.success is True
        assert result.content == "01234"
        assert result.data["line_count"] == 1
        assert result.data["line_count_complete"] is False
        assert result.data["truncated"] is True
        assert result.metadata["bytes"] == 10
        assert result.metadata["returned_chars"] == 5
        assert "chars" not in result.metadata
        assert read_sizes == [6]

    asyncio.run(run_test())


def test_list_files_uses_lookahead_for_accurate_bounded_results(
    tmp_path,
    monkeypatch,
):
    async def run_test() -> None:
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_bounded_list",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        tool = ListFilesTool()

        exact_call = ToolCall(
            call_id="call_exact_list",
            name=tool.name,
            arguments={"path": ".", "max_entries": 2},
        )
        exact = await tool.arun(
            tool.args_schema.model_validate(exact_call.arguments),
            ToolContext(context=context, call=exact_call),
        )

        assert exact.data["entries"] == ["a.txt", "b.txt"]
        assert exact.data["truncated"] is False

        consumed: list[int] = []

        def fake_entries(path, *, recursive):
            del recursive
            for index in range(100):
                consumed.append(index)
                yield path / f"entry-{index:03}.txt"

        monkeypatch.setattr(
            ListFilesTool,
            "_iter_entries",
            staticmethod(fake_entries),
        )
        limited_call = ToolCall(
            call_id="call_limited_list",
            name=tool.name,
            arguments={"path": ".", "max_entries": 2},
        )
        limited = await tool.arun(
            tool.args_schema.model_validate(limited_call.arguments),
            ToolContext(context=context, call=limited_call),
        )

        assert limited.data["entries"] == ["entry-000.txt", "entry-001.txt"]
        assert limited.data["truncated"] is True
        assert consumed == [0, 1, 2]

    asyncio.run(run_test())


def test_tool_runner_rejects_unknown_arguments_with_stable_error(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_strict_args",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        events = [
            event
            async for event in ToolRunner(ToolRegistry([ReadFileTool()])).run(
                ToolCall(
                    call_id="call_strict",
                    name="read_file",
                    arguments={"path": "README.md", "unexpected": True},
                ),
                context,
            )
        ]

        result = events[-1].payload["result"]
        assert result["error"] == "invalid_tool_arguments"
        assert result["metadata"]["validation_errors"][0]["location"] == "unexpected"

    asyncio.run(run_test())


def test_bash_command_risk_is_classified_once_by_tool_runner(tmp_path):
    class CountingPolicy(CommandPolicy):
        def __init__(self) -> None:
            self.calls = 0

        def classify(self, command: str, *, network_access: bool = False):
            self.calls += 1
            return super().classify(command, network_access=network_access)

    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_command_policy",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        policy = CountingPolicy()
        events = [
            event
            async for event in ToolRunner(
                ToolRegistry([BashTool()]),
                approval_manager=ApprovalManager(command_policy=policy),
            ).run(
                ToolCall(
                    call_id="call_pwd",
                    name="bash",
                    arguments={"command": "pwd"},
                ),
                context,
            )
        ]

        assert policy.calls == 1
        assert events[-1].payload["result"]["success"] is True

    asyncio.run(run_test())


def test_workspace_search_finds_paths_and_content_while_skipping_noise(tmp_path):
    async def run_test() -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (package / "agent.py").write_text(
            "def build_agent():\n    return 'workspace grounding'\n",
            encoding="utf-8",
        )
        noisy = tmp_path / "__pycache__"
        noisy.mkdir()
        (noisy / "hidden.py").write_text("workspace grounding\n", encoding="utf-8")
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )

        tool = WorkspaceSearchTool()
        call = ToolCall(
            call_id="call_search",
            name="workspace_search",
            arguments={"query": "agent", "path": "."},
        )

        result = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )

        assert result.success is True
        assert "pkg/agent.py [path]" in result.content
        assert "pkg/agent.py:1: def build_agent():" in result.content
        assert "__pycache__" not in result.content
        assert result.data["match_count"] == 2
        assert result.metadata["candidate_file_count"] == 1
        assert result.metadata["scanned_file_count"] == 1
        assert result.metadata["read_file_count"] == 1
        assert result.metadata["scanned_bytes"] > 0
        assert result.metadata["returned_chars"] == len(result.content)

    asyncio.run(run_test())


def test_workspace_search_rejects_path_escape(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        tool = WorkspaceSearchTool()
        call = ToolCall(
            call_id="call_search",
            name="workspace_search",
            arguments={"query": "secret", "path": "../outside"},
        )

        with pytest.raises(WorkspaceAccessError):
            await tool.arun(
                tool.args_schema.model_validate(call.arguments),
                ToolContext(context=context, call=call),
            )

    asyncio.run(run_test())


def test_write_file_tool_creates_and_updates_workspace_file(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        tool = WriteFileTool()
        call = ToolCall(
            call_id="call_write",
            name="write_file",
            arguments={
                "path": "notes/out.txt",
                "content": "hello",
                "create_parent_dirs": True,
            },
        )

        created = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )
        updated_call = ToolCall(
            call_id="call_write_2",
            name="write_file",
            arguments={"path": "notes/out.txt", "content": "hello again"},
        )
        updated = await tool.arun(
            tool.args_schema.model_validate(updated_call.arguments),
            ToolContext(context=context, call=updated_call),
        )

        assert (tmp_path / "notes" / "out.txt").read_text(
            encoding="utf-8"
        ) == "hello again"
        assert created.success is True
        assert created.data["status"] == "created"
        assert updated.success is True
        assert updated.data["status"] == "modified"
        assert "-hello" in updated.data["diff"]
        assert "+hello again" in updated.data["diff"]

    asyncio.run(run_test())


def test_write_file_tool_rejects_missing_parent_by_default(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        tool = WriteFileTool()
        call = ToolCall(
            call_id="call_write",
            name="write_file",
            arguments={"path": "missing/out.txt", "content": "hello"},
        )

        result = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )

        assert result.success is False
        assert result.error == "parent_directory_missing"
        assert not (tmp_path / "missing" / "out.txt").exists()

    asyncio.run(run_test())


def test_apply_patch_tool_modifies_workspace_file(tmp_path):
    async def run_test() -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        patch = """--- a/note.txt
+++ b/note.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+bravo
 gamma
"""
        tool = ApplyPatchTool()
        call = ToolCall(
            call_id="call_patch",
            name="apply_patch",
            arguments={"patch": patch},
        )

        result = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "alpha\nbravo\ngamma\n"
        assert result.data["modified"] == 1
        assert str(target) in result.data["changed_files"]

    asyncio.run(run_test())


def test_apply_patch_tool_treats_missing_transport_newline_as_complete_record():
    patch = """--- a/note.txt
+++ b/note.txt
@@ -1 +1 @@
-alpha
+bravo"""

    parsed = ApplyPatchTool._parse_patch(patch)
    result = ApplyPatchTool._apply_hunks("alpha\n", parsed[0].hunks)

    assert result == "bravo\n"


def test_apply_patch_tool_respects_explicit_no_newline_marker():
    patch = """--- a/note.txt
+++ b/note.txt
@@ -1 +1 @@
-alpha
+bravo
\\ No newline at end of file"""

    parsed = ApplyPatchTool._parse_patch(patch)
    result = ApplyPatchTool._apply_hunks("alpha\n", parsed[0].hunks)

    assert result == "bravo"


def test_apply_patch_tool_rejects_workspace_escape(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        patch = """--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""
        tool = ApplyPatchTool()
        call = ToolCall(
            call_id="call_patch",
            name="apply_patch",
            arguments={"patch": patch},
        )

        result = await tool.arun(
            tool.args_schema.model_validate(call.arguments),
            ToolContext(context=context, call=call),
        )

        assert result.success is False
        assert result.error == "workspace_access_denied"

    asyncio.run(run_test())


def test_bash_tool_runs_safe_command_and_blocks_prompt_or_denied(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        tool = BashTool()
        safe_call = ToolCall(
            call_id="call_bash",
            name="bash",
            arguments={"command": "pwd"},
        )
        prompt_call = ToolCall(
            call_id="call_rm",
            name="bash",
            arguments={"command": "rm file.txt"},
        )
        denied_call = ToolCall(
            call_id="call_sudo",
            name="bash",
            arguments={"command": "sudo true"},
        )

        safe = await tool.arun(
            tool.args_schema.model_validate(safe_call.arguments),
            ToolContext(
                context=context,
                call=safe_call,
                command_decision=CommandPolicy().classify("pwd"),
            ),
        )
        prompt = await tool.arun(
            tool.args_schema.model_validate(prompt_call.arguments),
            ToolContext(
                context=context,
                call=prompt_call,
                command_decision=CommandPolicy().classify("rm file.txt"),
            ),
        )
        denied = await tool.arun(
            tool.args_schema.model_validate(denied_call.arguments),
            ToolContext(
                context=context,
                call=denied_call,
                command_decision=CommandPolicy().classify("sudo true"),
            ),
        )

        assert safe.success is True
        assert safe.data["exit_code"] == 0
        assert str(tmp_path) in safe.content
        assert prompt.success is False
        assert prompt.error == "command_requires_approval"
        assert denied.success is False
        assert denied.error == "command_denied"

    asyncio.run(run_test())


def test_sandbox_policy_denies_side_effects_in_read_only():
    policy = SandboxPolicy(
        mode=SandboxMode.READ_ONLY,
        network_access=False,
    )

    read = policy.evaluate_effects({ToolEffect.READ_ONLY})
    write = policy.evaluate_effects({ToolEffect.WORKSPACE_WRITE})
    network = policy.evaluate_effects({ToolEffect.NETWORK})

    assert read.allowed is True
    assert write.allowed is False
    assert write.denied_effect == ToolEffect.WORKSPACE_WRITE
    assert network.allowed is False
    assert network.denied_effect == ToolEffect.NETWORK


def test_tool_runner_denies_workspace_write_in_read_only_sandbox(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"sandbox_mode": "read_only"})
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        reviewer = AutoApprovalReviewer(approved=True)
        runner = ToolRunner(
            ToolRegistry([WriteFileTool()]),
            approval_manager=ApprovalManager(
                reviewer=reviewer,
            ),
        )

        events = [
            event
            async for event in runner.run(
                ToolCall(
                    call_id="call_write",
                    name="write_file",
                    arguments={"path": "blocked.txt", "content": "nope"},
                ),
                context,
            )
        ]

        assert not (tmp_path / "blocked.txt").exists()
        assert [event.type for event in events] == [
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
        ]
        assert reviewer.requests == []
        assert events[1].payload["result"]["error"] == "sandbox_denied"
        assert (
            events[1].payload["result"]["metadata"]["denied_effect"]
            == "workspace_write"
        )

    asyncio.run(run_test())


def test_tool_runner_denies_bash_in_read_only_sandbox(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path).model_copy(update={"sandbox_mode": "read_only"})
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model=config.model,
            model_provider=config.model_provider,
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        runner = ToolRunner(
            ToolRegistry([BashTool()]),
            approval_manager=ApprovalManager(),
        )

        events = [
            event
            async for event in runner.run(
                ToolCall(
                    call_id="call_bash",
                    name="bash",
                    arguments={"command": "pwd"},
                ),
                context,
            )
        ]

        assert [event.type for event in events] == [
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
        ]
        assert events[1].payload["result"]["error"] == "sandbox_denied"
        assert (
            events[1].payload["result"]["metadata"]["denied_effect"] == "process_exec"
        )

    asyncio.run(run_test())


def test_session_config_normalizes_paths(tmp_path):
    config = make_config(tmp_path)

    assert config.cwd == tmp_path.resolve()
    assert config.codecraft_home == (tmp_path / ".codecraft").resolve()


def test_session_config_requires_known_policy_names(tmp_path):
    config_data = make_config(tmp_path).model_dump()
    config = SessionConfig.model_validate(config_data)

    assert config.approval_policy == ApprovalPolicy.NEVER
    assert config.sandbox_mode == SandboxMode.WORKSPACE_WRITE
    assert config.model_dump(mode="json")["approval_policy"] == "never"
    assert config.model_dump(mode="json")["sandbox_mode"] == "workspace_write"

    with pytest.raises(ValueError, match="approval_policy"):
        SessionConfig.model_validate({**config_data, "approval_policy": "sometimes"})

    with pytest.raises(ValueError, match="sandbox_mode"):
        SessionConfig.model_validate({**config_data, "sandbox_mode": "half_trusted"})


def test_session_config_rejects_stale_fields_and_invalid_budgets(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_data = make_config(workspace).model_dump(mode="python")

    with pytest.raises(ValueError, match="thread_id"):
        SessionConfig.model_validate({**config_data, "thread_id": "thr_stale"})
    with pytest.raises(ValueError, match="project_instructions"):
        SessionConfig.model_validate(
            {**config_data, "project_instructions": "stale snapshot"}
        )
    with pytest.raises(ValueError, match="workspace_roots"):
        SessionConfig.model_validate({**config_data, "workspace_roots": [workspace]})
    with pytest.raises(ValueError, match="max_tool_calls"):
        SessionConfig.model_validate({**config_data, "max_tool_calls": 0})
    with pytest.raises(ValueError, match="max_tool_output_chars"):
        SessionConfig.model_validate({**config_data, "max_tool_output_chars": 0})
    with pytest.raises(ValueError, match="model output tokens"):
        SessionConfig.model_validate(
            {
                **config_data,
                "model_context_window_tokens": 4096,
                "model_max_output_tokens": 4096,
            }
        )
    with pytest.raises(ValueError, match="model_api_key_env"):
        SessionConfig.model_validate({**config_data, "model_api_key_env": "BAD-NAME"})
    with pytest.raises(ValueError, match="MCP server names"):
        SessionConfig.model_validate(
            {
                **config_data,
                "mcp_servers": {"bad server": {"command": "python"}},
            }
        )


def test_turn_context_is_immutable(tmp_path):
    config = make_config(tmp_path)
    context = TurnContext(
        session_id=config.session_id,
        turn_id="turn_test",
        cwd=config.cwd,
        model=config.model,
        model_provider=config.model_provider,
        approval_policy=config.approval_policy,
        sandbox_mode=config.sandbox_mode,
        network_access=config.network_access,
        available_tools=[
            ToolSpec(
                name="read_file",
                description="Read a workspace file.",
                input_schema={"type": "object"},
                effects={ToolEffect.READ_ONLY},
            )
        ],
        max_tool_calls=config.max_tool_calls,
        max_tool_output_chars=config.max_tool_output_chars,
        created_at=config.created_at,
    )

    with pytest.raises(ValueError):
        context.turn_id = "turn_other"


def test_llm_provider_stream_contract(tmp_path):
    class MockProvider(LLMProvider):
        name = "mock"

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelEvent]:
            yield ModelEvent(
                type=ModelEventType.MESSAGE_COMPLETED,
                payload={"text": request.messages[0].content},
            )
            yield ModelEvent(type=ModelEventType.COMPLETED)

    config = make_config(tmp_path)
    context = TurnContext(
        session_id=config.session_id,
        turn_id="turn_test",
        cwd=config.cwd,
        model=config.model,
        model_provider=config.model_provider,
        approval_policy=config.approval_policy,
        sandbox_mode=config.sandbox_mode,
        network_access=config.network_access,
        available_tools=[],
        max_tool_calls=config.max_tool_calls,
        max_tool_output_chars=config.max_tool_output_chars,
        created_at=config.created_at,
    )

    async def collect() -> list[ModelEvent]:
        provider = MockProvider()
        return [
            event
            async for event in provider.stream(
                ModelRequest(
                    model=context.model,
                    messages=(ModelMessage(role=ModelRole.USER, content="hello"),),
                )
            )
        ]

    events = asyncio.run(collect())

    assert [event.type for event in events] == [
        ModelEventType.MESSAGE_COMPLETED,
        ModelEventType.COMPLETED,
    ]
    assert events[0].payload.text == "hello"


def test_openai_provider_converts_response_to_model_events(tmp_path):
    class FakeResponses:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": '{"path": "README.md"}',
                    },
                    {
                        "type": "message",
                        "content": [{"text": "done"}],
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            }

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model="gpt-test",
            model_provider="openai",
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[
                ToolSpec(
                    name="read_file",
                    description="Read file.",
                    input_schema={"type": "object"},
                )
            ],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        client = FakeClient()
        provider = OpenAIProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="gpt-test",
                    messages=(ModelMessage(role=ModelRole.USER, content="read"),),
                    tools=tuple(context.available_tools),
                )
            )
        ]

        assert client.responses.kwargs["model"] == "gpt-test"
        assert client.responses.kwargs["input"] == [{"role": "user", "content": "read"}]
        assert client.responses.kwargs["stream"] is True
        assert client.responses.kwargs["store"] is False
        assert client.responses.kwargs["max_output_tokens"] == 8192
        assert client.responses.kwargs["tools"][0]["name"] == "read_file"
        assert [event.type for event in events] == [
            ModelEventType.MESSAGE_COMPLETED,
            ModelEventType.TOKEN_COUNT,
            ModelEventType.TOOL_CALL,
            ModelEventType.COMPLETED,
        ]
        assert events[0].payload.text == "done"
        assert events[1].payload.total_tokens == 15
        assert events[1].payload.reasoning_tokens == 2
        assert events[1].payload.cached_input_tokens == 3
        assert events[2].payload.arguments == {"path": "README.md"}

    asyncio.run(run_test())


def test_openai_provider_streams_response_deltas_and_tool_calls(tmp_path):
    class FakeStream:
        def __aiter__(self):
            self.events = iter(
                [
                    {
                        "type": "response.output_text.delta",
                        "delta": "hello ",
                    },
                    {
                        "type": "response.output_text.delta",
                        "delta": "stream",
                    },
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "function_call",
                            "call_id": "call_read",
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            "output_text": "hello stream",
                            "usage": {
                                "input_tokens": 2,
                                "output_tokens": 3,
                            },
                        },
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.events)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeResponses:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeStream()

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    async def run_test() -> None:
        client = FakeClient()
        provider = OpenAIProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="gpt-test",
                    messages=(ModelMessage(role=ModelRole.USER, content="read"),),
                )
            )
        ]

        assert client.responses.kwargs["stream"] is True
        assert [event.type for event in events] == [
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.TOKEN_COUNT,
            ModelEventType.TOOL_CALL,
            ModelEventType.COMPLETED,
        ]
        assert [event.payload.text for event in events[:2]] == [
            "hello ",
            "stream",
        ]
        assert events[2].payload.total_tokens == 5
        assert events[3].payload.arguments == {"path": "README.md"}

    asyncio.run(run_test())


def test_openai_provider_serializes_tool_history_as_response_items(tmp_path):
    class FakeResponses:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return {"output_text": "final"}

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    async def run_test() -> None:
        client = FakeClient()
        provider = OpenAIProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="gpt-test",
                    messages=(
                        ModelMessage(role=ModelRole.USER, content="read README"),
                        ModelMessage(
                            type=ModelMessageType.TOOL_CALL,
                            role=ModelRole.ASSISTANT,
                            name="read_file",
                            tool_call_id="call_read",
                            arguments={"path": "README.md"},
                        ),
                        ModelMessage(
                            type=ModelMessageType.TOOL_RESULT,
                            role=ModelRole.TOOL,
                            content="README contents",
                            tool_call_id="call_read",
                        ),
                    ),
                )
            )
        ]

        assert client.responses.kwargs["input"] == [
            {"role": "user", "content": "read README"},
            {
                "type": "function_call",
                "call_id": "call_read",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": "README contents",
            },
        ]
        assert events[0].payload.text == "final"

    asyncio.run(run_test())


def test_qwen_provider_streams_chat_completion_deltas(tmp_path):
    class FakeStream:
        def __aiter__(self):
            self.chunks = iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "qwen ",
                                }
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "stream",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 4,
                            "total_tokens": 7,
                        },
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeStream()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    async def run_test() -> None:
        client = FakeClient()
        provider = QwenProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="qwen-plus",
                    messages=(ModelMessage(role=ModelRole.USER, content="hello"),),
                )
            )
        ]

        assert client.chat.completions.kwargs["model"] == "qwen-plus"
        assert client.chat.completions.kwargs["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert client.chat.completions.kwargs["stream"] is True
        assert client.chat.completions.kwargs["max_tokens"] == 8192
        assert "tools" not in client.chat.completions.kwargs
        assert [event.type for event in events] == [
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.TOKEN_COUNT,
            ModelEventType.COMPLETED,
        ]
        assert [event.payload.text for event in events[:2]] == [
            "qwen ",
            "stream",
        ]
        assert events[2].payload.total_tokens == 7

    asyncio.run(run_test())


def test_qwen_provider_streams_chat_completion_tool_calls(tmp_path):
    class FakeStream:
        def __aiter__(self):
            self.chunks = iter(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_read",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": '"README.md"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeStream()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    async def run_test() -> None:
        config = make_config(tmp_path)
        context = TurnContext(
            session_id=config.session_id,
            turn_id="turn_test",
            cwd=config.cwd,
            model="qwen-plus",
            model_provider="qwen",
            approval_policy=config.approval_policy,
            sandbox_mode=config.sandbox_mode,
            network_access=config.network_access,
            available_tools=[
                ToolSpec(
                    name="read_file",
                    description="Read file.",
                    input_schema={"type": "object"},
                )
            ],
            max_tool_calls=config.max_tool_calls,
            max_tool_output_chars=config.max_tool_output_chars,
            created_at=config.created_at,
        )
        client = FakeClient()
        provider = QwenProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="qwen-plus",
                    messages=(
                        ModelMessage(role=ModelRole.USER, content="read README"),
                        ModelMessage(
                            type=ModelMessageType.TOOL_CALL,
                            role=ModelRole.ASSISTANT,
                            name="read_file",
                            tool_call_id="call_previous",
                            arguments={"path": "README.md"},
                        ),
                        ModelMessage(
                            type=ModelMessageType.TOOL_RESULT,
                            role=ModelRole.TOOL,
                            content="README contents",
                            tool_call_id="call_previous",
                        ),
                    ),
                    tools=tuple(context.available_tools),
                )
            )
        ]

        assert client.chat.completions.kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file.",
                    "parameters": {"type": "object"},
                },
            }
        ]
        assert client.chat.completions.kwargs["messages"][1:] == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_previous",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_previous",
                "content": "README contents",
            },
        ]
        assert [event.type for event in events] == [
            ModelEventType.TOOL_CALL,
            ModelEventType.COMPLETED,
        ]
        assert events[0].payload.model_dump(mode="json") == {
            "call_id": "call_read",
            "name": "read_file",
            "arguments": {"path": "README.md"},
        }

    asyncio.run(run_test())


def test_responses_and_chat_providers_have_separate_protocol_bases():
    assert not isinstance(QwenProvider(client=object()), OpenAIProvider)
    assert not isinstance(DeepSeekProvider(client=object()), OpenAIProvider)


def test_deepseek_provider_streams_chat_completion_deltas(tmp_path):
    class FakeStream:
        def __aiter__(self):
            self.chunks = iter(
                [
                    {"choices": [{"delta": {"content": "deepseek "}}]},
                    {
                        "choices": [
                            {
                                "delta": {"content": "stream"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 6,
                            "total_tokens": 11,
                        },
                    },
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self) -> None:
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeStream()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    async def run_test() -> None:
        client = FakeClient()
        provider = DeepSeekProvider(client=client)

        events = [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="deepseek-v4-flash",
                    messages=(ModelMessage(role=ModelRole.USER, content="hello"),),
                )
            )
        ]

        assert client.chat.completions.kwargs["model"] == "deepseek-v4-flash"
        assert client.chat.completions.kwargs["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert client.chat.completions.kwargs["stream"] is True
        assert [event.type for event in events] == [
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.MESSAGE_DELTA,
            ModelEventType.TOKEN_COUNT,
            ModelEventType.COMPLETED,
        ]
        assert [event.payload.text for event in events[:2]] == [
            "deepseek ",
            "stream",
        ]
        assert events[2].payload.total_tokens == 11

    asyncio.run(run_test())


def test_qwen_provider_requires_dashscope_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    provider = QwenProvider()

    with pytest.raises(LLMConfigError, match="DASHSCOPE_API_KEY"):
        provider._client()


def test_deepseek_provider_requires_deepseek_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = DeepSeekProvider()

    with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
        provider._client()


def test_session_store_appends_loads_lists_and_resumes(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        path = await store.create_session(config)

        assert path.exists()

        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                seq=1,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_test",
                seq=2,
                type=RuntimeEventType.TURN_STARTED,
                payload={"input_id": "inp_test"},
            )
        )

        events = await store.load_events(config.session_id)
        summaries = await store.list_sessions(cwd=tmp_path)
        snapshot = await store.resume_last(cwd=tmp_path)

        assert [event.seq for event in events] == [1, 2]
        assert summaries[0].session_id == config.session_id
        assert summaries[0].event_count == 2
        assert snapshot.config.session_id == config.session_id
        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
        ]

    asyncio.run(run_test())


def test_session_store_rejects_seq_gaps(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        await store.create_session(config)
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                seq=2,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )

        with pytest.raises(Exception, match="sequence"):
            await store.load_events(config.session_id)

    asyncio.run(run_test())


def test_session_store_rejects_unknown_event_schema_version(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        path = await store.create_session(config)
        event = RuntimeEvent(
            event_id="evt_version",
            session_id=config.session_id,
            seq=1,
            type=RuntimeEventType.SESSION_STARTED,
            payload={"config": config.model_dump(mode="json")},
        ).model_dump(mode="json")
        event["schema_version"] = 2
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with pytest.raises(SessionRestoreError) as raised:
            await store.load_events(config.session_id)

        assert raised.value.code == "session_event_schema_unsupported"
        assert raised.value.metadata["version"] == 2

    asyncio.run(run_test())


def test_session_store_rejects_unknown_config_schema_version(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        path = await store.create_session(config)
        event = RuntimeEvent(
            event_id="evt_started",
            session_id=config.session_id,
            seq=1,
            type=RuntimeEventType.SESSION_STARTED,
            payload={"config": config.model_dump(mode="json")},
        ).model_dump(mode="json")
        event["payload"]["config"]["schema_version"] = 2
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with pytest.raises(SessionRestoreError) as raised:
            await store.resume(config.session_id)

        assert raised.value.code == "session_config_schema_unsupported"
        assert raised.value.metadata["version"] == 2

    asyncio.run(run_test())


def test_session_store_rejects_missing_schema_versions(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        path = await store.create_session(config)
        event = RuntimeEvent(
            event_id="evt_started",
            session_id=config.session_id,
            seq=1,
            type=RuntimeEventType.SESSION_STARTED,
            payload={"config": config.model_dump(mode="json")},
        ).model_dump(mode="json")
        event["payload"]["config"].pop("schema_version")
        event.pop("schema_version")
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with pytest.raises(SessionRestoreError) as raised:
            await store.resume(config.session_id)

        assert raised.value.code == "session_event_schema_unsupported"
        assert raised.value.metadata["version"] is None

        event["schema_version"] = 1
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")

        with pytest.raises(SessionRestoreError) as raised:
            await store.resume(config.session_id)

        assert raised.value.code == "session_config_schema_unsupported"
        assert raised.value.metadata["version"] is None

    asyncio.run(run_test())


def test_session_list_and_resume_share_restorability_validation(tmp_path):
    async def run_test() -> None:
        store = SessionStore(tmp_path / ".codecraft")
        expected_codes = {
            "ses_empty": "session_empty",
            "ses_wrong_start": "session_start_event_missing",
            "ses_missing_config": "session_config_missing",
            "ses_invalid_config": "session_config_validation_failed",
            "ses_config_id_mismatch": "session_config_id_mismatch",
        }

        for session_id in expected_codes:
            config = make_config(tmp_path).model_copy(update={"session_id": session_id})
            path = await store.create_session(config)
            if session_id == "ses_empty":
                continue
            if session_id == "ses_wrong_start":
                event_data = RuntimeEvent(
                    event_id=f"evt_{session_id}",
                    session_id=session_id,
                    seq=1,
                    type=RuntimeEventType.SESSION_CLOSED,
                ).model_dump(mode="json")
            else:
                event_data = RuntimeEvent(
                    event_id=f"evt_{session_id}",
                    session_id=session_id,
                    seq=1,
                    type=RuntimeEventType.SESSION_STARTED,
                    payload={"config": config.model_dump(mode="json")},
                ).model_dump(mode="json")
                if session_id == "ses_missing_config":
                    event_data["payload"] = {}
                elif session_id == "ses_invalid_config":
                    event_data["payload"]["config"].pop("model")
                else:
                    event_data["payload"]["config"]["session_id"] = "ses_other"
            path.write_text(json.dumps(event_data) + "\n", encoding="utf-8")

        assert await store.load_events("ses_empty") == []
        wrong_start = await store.load_events("ses_wrong_start")
        assert wrong_start[0].type == RuntimeEventType.SESSION_CLOSED

        resume_codes = {}
        for session_id in expected_codes:
            with pytest.raises(SessionRestoreError) as raised:
                await store.resume(session_id)
            resume_codes[session_id] = raised.value.code

        assert resume_codes == expected_codes
        assert await store.list_sessions() == []
        all_summaries = await store.list_sessions(include_invalid=True)
        assert {
            summary.session_id: summary.error_code for summary in all_summaries
        } == expected_codes

    asyncio.run(run_test())


def test_session_list_streams_summaries_without_loading_event_lists(tmp_path):
    class SummaryOnlyStore(SessionStore):
        async def load_events(self, session_id: str) -> list[RuntimeEvent]:
            raise AssertionError(f"unexpected full event load for {session_id}")

    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SummaryOnlyStore(config.codecraft_home)
        await store.create_session(config)
        await store.append_event(
            RuntimeEvent(
                event_id="evt_started",
                session_id=config.session_id,
                seq=1,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )

        summaries = await store.list_sessions(cwd=tmp_path)

        assert len(summaries) == 1
        assert summaries[0].event_count == 1

    asyncio.run(run_test())


def test_session_list_isolates_invalid_utf8_logs(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        path = await store.create_session(config)
        path.write_bytes(b"\xff\xfe\n")

        with pytest.raises(SessionRestoreError) as raised:
            await store.resume(config.session_id)

        assert raised.value.code == "session_events_load_failed"
        assert await store.list_sessions() == []
        summaries = await store.list_sessions(include_invalid=True)
        assert len(summaries) == 1
        assert summaries[0].error_code == "session_events_load_failed"
        assert summaries[0].event_count == 0

    asyncio.run(run_test())


def test_resume_last_falls_back_when_a_scanned_session_becomes_invalid(tmp_path):
    class RacingSessionStore(SessionStore):
        def __init__(self, codecraft_home: Path) -> None:
            super().__init__(codecraft_home)
            self.fail_ids: set[str] = set()
            self.resume_calls: list[str] = []

        async def resume(self, session_id: str) -> SessionSnapshot:
            self.resume_calls.append(session_id)
            if session_id in self.fail_ids:
                raise SessionRestoreError(
                    "session changed after scan",
                    code="session_changed_after_scan",
                )
            return await super().resume(session_id)

    async def seed(
        store: SessionStore,
        session_id: str,
        timestamp: datetime,
    ) -> None:
        config = make_config(tmp_path).model_copy(update={"session_id": session_id})
        await store.create_session(config)
        await store.append_event(
            RuntimeEvent(
                event_id=f"evt_{session_id}",
                session_id=session_id,
                seq=1,
                timestamp=timestamp,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )

    async def run_test() -> None:
        store = RacingSessionStore(tmp_path / ".codecraft")
        now = datetime.now(UTC)
        await seed(store, "ses_older", now)
        await seed(store, "ses_newer", now + timedelta(seconds=1))
        store.fail_ids = {"ses_newer"}

        snapshot = await store.resume_last(cwd=tmp_path)

        assert snapshot.config.session_id == "ses_older"
        assert store.resume_calls == ["ses_newer", "ses_older"]

        store.resume_calls.clear()
        store.fail_ids = {"ses_newer", "ses_older"}
        with pytest.raises(SessionRestoreError) as raised:
            await store.resume_last(cwd=tmp_path)

        assert raised.value.code == "session_not_found"
        assert raised.value.metadata["attempts"] == [
            {"session_id": "ses_newer", "code": "session_changed_after_scan"},
            {"session_id": "ses_older", "code": "session_changed_after_scan"},
        ]

    asyncio.run(run_test())


def test_session_store_list_skips_invalid_session_logs(tmp_path):
    async def run_test() -> None:
        bad_config = make_config(tmp_path).model_copy(update={"session_id": "ses_bad"})
        good_config = make_config(tmp_path).model_copy(
            update={"session_id": "ses_good"}
        )
        store = SessionStore(bad_config.codecraft_home)

        await store.create_session(bad_config)
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=bad_config.session_id,
                seq=2,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": bad_config.model_dump(mode="json")},
            )
        )

        await store.create_session(good_config)
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=good_config.session_id,
                seq=1,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": good_config.model_dump(mode="json")},
            )
        )

        summaries = await store.list_sessions(cwd=tmp_path)
        all_summaries = await store.list_sessions(include_invalid=True)
        snapshot = await store.resume_last(cwd=tmp_path)

        assert [summary.session_id for summary in summaries] == ["ses_good"]
        assert {summary.session_id for summary in all_summaries} == {
            "ses_bad",
            "ses_good",
        }
        invalid = next(
            summary for summary in all_summaries if summary.session_id == "ses_bad"
        )
        assert invalid.valid is False
        assert invalid.error_code == "session_seq_not_continuous"
        assert invalid.event_count == 1
        assert snapshot.config.session_id == "ses_good"

    asyncio.run(run_test())


def test_agent_runtime_creates_thread_and_runs_basic_turn(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "hello "},
                ),
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "runtime"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "say hello"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        assert [event.seq for event in snapshot.events] == list(range(1, 8))
        assert snapshot.events[-1].payload["answer"] == "hello runtime"
        assert provider.calls[0].messages[0].role == ModelRole.SYSTEM
        assert provider.calls[0].messages[1].content == "say hello"

    asyncio.run(run_test())


def test_agent_thread_next_event_sees_session_started_and_turn_events(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        script=[
                            ModelEvent(
                                type=ModelEventType.MESSAGE_COMPLETED,
                                payload={"text": "done"},
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

        thread = await runtime.create_thread(config)
        first = await thread.next_event()
        await thread.submit(SessionInput.user_message("inp_test", "finish"))
        await thread.wait_until_idle()
        second = await thread.next_event()
        third = await thread.next_event()

        assert first.type == RuntimeEventType.SESSION_STARTED
        assert second.type == RuntimeEventType.TURN_STARTED
        assert third.type == RuntimeEventType.USER_MESSAGE

    asyncio.run(run_test())


def test_runtime_resume_reconstructs_conversation_without_replaying_turn(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        first_provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "first answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        first_runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([first_provider]),
            tool_registry=ToolRegistry(),
        )
        thread = await first_runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "first"))
        await thread.wait_until_idle()

        second_provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "second answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        second_runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([second_provider]),
            tool_registry=ToolRegistry(),
        )
        resumed = await second_runtime.resume_thread(config.session_id)
        restored = await resumed.next_event()
        await resumed.submit(SessionInput.user_message("inp_two", "second"))
        await resumed.wait_until_idle()

        assert restored.type == RuntimeEventType.SESSION_RESTORED
        assert len(first_provider.calls) == 1
        assert len(second_provider.calls) == 1
        assert [
            message.content for message in second_provider.calls[0].messages[1:]
        ] == [
            "first",
            "first answer",
            "second",
        ]

    asyncio.run(run_test())


def test_runtime_resume_last_reuses_selected_snapshot(tmp_path):
    class CountingSessionStore(SessionStore):
        def __init__(self, codecraft_home: Path) -> None:
            super().__init__(codecraft_home)
            self.resume_calls = 0

        async def resume(self, session_id: str) -> SessionSnapshot:
            self.resume_calls += 1
            return await super().resume(session_id)

    async def run_test() -> None:
        config = make_config(tmp_path)
        store = CountingSessionStore(config.codecraft_home)
        await store.create_session(config)
        await store.append_event(
            RuntimeEvent(
                event_id="evt_started",
                session_id=config.session_id,
                seq=1,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )
        runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([MockProvider()]),
            tool_registry=ToolRegistry(),
        )

        thread = await runtime.resume_last(cwd=tmp_path)
        restored = await thread.next_event()

        assert restored.type == RuntimeEventType.SESSION_RESTORED
        assert store.resume_calls == 1
        assert thread.session.conversation.items == []

        await runtime.close()

    asyncio.run(run_test())


def test_runtime_resume_reconstructs_tool_call_and_result_history(tmp_path):
    async def run_test() -> None:
        (tmp_path / "note.txt").write_text("resume sees tool result", encoding="utf-8")
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        first_provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "note.txt"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "first answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        first_runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([first_provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )
        thread = await first_runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "read note"))
        await thread.wait_until_idle()

        second_provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "second answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        second_runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([second_provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )
        resumed = await second_runtime.resume_thread(config.session_id)
        await resumed.next_event()
        await resumed.submit(SessionInput.user_message("inp_two", "continue"))
        await resumed.wait_until_idle()

        assert len(first_provider.calls) == 2
        assert len(second_provider.calls) == 1
        messages = second_provider.calls[0].messages
        assert [message.content for message in messages[1:]] == [
            "read note",
            None,
            "resume sees tool result",
            "first answer",
            "continue",
        ]
        assert [message.role.value for message in messages[1:]] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "user",
        ]
        assert messages[2].type == ModelMessageType.TOOL_CALL
        assert messages[2].arguments == {"path": "note.txt"}
        assert messages[3].type == ModelMessageType.TOOL_RESULT

    asyncio.run(run_test())


def test_runtime_injects_system_instructions_before_conversation(tmp_path):
    async def run_test() -> None:
        (tmp_path / "AGENTS.md").write_text(
            "Project rule: inspect files first.", encoding="utf-8"
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "answer"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path).model_copy(
            update={"user_instructions": "User rule: answer briefly."}
        )
        (tmp_path / "AGENTS.md").write_text(
            "Changed after session creation.", encoding="utf-8"
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "hello"))
        await thread.wait_until_idle()

        messages = provider.calls[0].messages
        assert messages[0].role == ModelRole.SYSTEM
        assert "<base_instructions>" in messages[0].content
        assert "Project rule: inspect files first." not in messages[0].content
        assert "Changed after session creation." in messages[0].content
        assert "User rule: answer briefly." in messages[0].content
        assert "approval_policy: never" in messages[0].content
        assert messages[1].role == ModelRole.USER
        assert messages[1].content == "hello"

    asyncio.run(run_test())


def test_runtime_loads_scoped_instructions_after_accessing_nested_path(tmp_path):
    async def run_test() -> None:
        package = tmp_path / "pkg"
        package.mkdir()
        (tmp_path / "AGENTS.md").write_text("root rule", encoding="utf-8")
        (package / "AGENTS.md").write_text("package rule", encoding="utf-8")
        (package / "note.txt").write_text("hello", encoding="utf-8")
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "pkg/note.txt"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "done"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_one", "read note"))
        await thread.wait_until_idle()

        assert "root rule" in provider.calls[0].messages[0].content
        assert "package rule" not in provider.calls[0].messages[0].content
        assert "package rule" in provider.calls[1].messages[0].content
        assert "scope: pkg" in provider.calls[1].messages[0].content

    asyncio.run(run_test())


def test_runtime_resume_uses_context_compaction_summary(tmp_path):
    async def run_test() -> None:
        config = make_config(tmp_path)
        store = SessionStore(config.codecraft_home)
        await store.create_session(config)
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                seq=1,
                type=RuntimeEventType.SESSION_STARTED,
                payload={"config": config.model_dump(mode="json")},
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_one",
                seq=2,
                type=RuntimeEventType.USER_MESSAGE,
                payload={"input_id": "inp_one", "text": "old user"},
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_one",
                seq=3,
                type=RuntimeEventType.CONTEXT_COMPACTED,
                payload={
                    "summary": "old conversation summary",
                    "before_tokens": 10,
                    "after_tokens": 5,
                    "removed_items": 1,
                    "retained_items": 0,
                    "conversation": {
                        "items": [
                            {
                                "item_id": "item_summary",
                                "role": "summary",
                                "content": "old conversation summary",
                            }
                        ]
                    },
                },
            )
        )
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "after compact"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        runtime = AgentRuntime(
            session_store=store,
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )
        resumed = await runtime.resume_thread(config.session_id)
        await resumed.next_event()
        await resumed.submit(SessionInput.user_message("inp_two", "new user"))
        await resumed.wait_until_idle()

        messages = provider.calls[0].messages
        assert "<base_instructions>" in messages[0].content
        assert [message.content for message in messages[1:]] == [
            "old conversation summary",
            "new user",
        ]
        assert [message.role.value for message in messages[1:]] == [
            "user",
            "user",
        ]

    asyncio.run(run_test())


def test_runtime_allows_final_answer_after_reaching_tool_call_limit(tmp_path):
    async def run_test() -> None:
        (tmp_path / "note.txt").write_text("tool loop works", encoding="utf-8")
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "note.txt"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "The file says: tool loop works"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path).model_copy(update={"max_tool_calls": 1})
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "read note"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        finished = snapshot.events[5]
        assert finished.payload["result"]["success"] is True
        assert finished.payload["result"]["content"] == "tool loop works"
        assert snapshot.events[-1].payload["tool_calls"] == 1
        messages = provider.calls[1].messages
        assert [message.content for message in messages[1:]] == [
            "read note",
            None,
            "tool loop works",
        ]
        assert messages[2].type == ModelMessageType.TOOL_CALL
        assert messages[3].type == ModelMessageType.TOOL_RESULT

    asyncio.run(run_test())


def test_runtime_preserves_streamed_assistant_text_before_tool_call(tmp_path):
    async def run_test() -> None:
        (tmp_path / "note.txt").write_text("tool loop works", encoding="utf-8")
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "I will "},
                ),
                ModelEvent(
                    type=ModelEventType.MESSAGE_DELTA,
                    payload={"text": "read that."},
                ),
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_read",
                        "name": "read_file",
                        "arguments": {"path": "note.txt"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "The file says: tool loop works"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ReadFileTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "read note"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            RuntimeEventType.ASSISTANT_MESSAGE_DELTA,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        assert snapshot.events[5].payload["text"] == "I will read that."

        messages = provider.calls[1].messages
        assert [message.content for message in messages[1:]] == [
            "read note",
            "I will read that.",
            None,
            "tool loop works",
        ]
        assert [message.role.value for message in messages[1:]] == [
            "user",
            "assistant",
            "assistant",
            "tool",
        ]

    asyncio.run(run_test())


def test_runtime_records_failed_unknown_tool(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_missing",
                        "name": "missing_tool",
                        "arguments": {},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Missing tool was reported."},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry(),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "call missing"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        finished = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        ][0]
        assert finished.payload["result"]["success"] is False
        assert finished.payload["result"]["error"] == "tool_not_found"
        assert "[tool_error: tool_not_found]" in provider.calls[1].messages[-1].content
        assert snapshot.events[-1].type == RuntimeEventType.TURN_FINISHED

    asyncio.run(run_test())


def test_runtime_executes_write_file_tool_call(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {
                            "path": "generated.txt",
                            "content": "created by runtime",
                        },
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Wrote generated.txt"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([WriteFileTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "write file"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert (tmp_path / "generated.txt").read_text(
            encoding="utf-8"
        ) == "created by runtime"
        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        finished = snapshot.events[5]
        assert finished.payload["result"]["success"] is True
        assert finished.payload["result"]["data"]["status"] == "created"
        assert provider.calls[1].messages[-1].content.startswith("created ")

    asyncio.run(run_test())


def test_runtime_emits_patch_applied_event(tmp_path):
    async def run_test() -> None:
        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        patch = """--- a/note.txt
+++ b/note.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+bravo
 gamma
"""
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_patch",
                        "name": "apply_patch",
                        "arguments": {"patch": patch},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Patched note.txt"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([ApplyPatchTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "patch file"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert target.read_text(encoding="utf-8") == "alpha\nbravo\ngamma\n"
        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.TOOL_CALL_FINISHED,
            RuntimeEventType.PATCH_APPLIED,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        patch_event = snapshot.events[6]
        assert patch_event.payload["modified"] == 1
        assert str(target) in patch_event.payload["changed_files"]

    asyncio.run(run_test())


def test_runtime_executes_bash_tool_call(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_bash",
                        "name": "bash",
                        "arguments": {"command": "pwd"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Ran pwd"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path)
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([BashTool()]),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "run pwd"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        finished = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        ][0]
        assert finished.payload["result"]["success"] is True
        assert str(tmp_path) in finished.payload["result"]["data"]["stdout"]
        assert snapshot.events[-1].type == RuntimeEventType.TURN_FINISHED

    asyncio.run(run_test())


def test_tool_runner_emits_approval_events_and_runs_approved_prompt_command(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_bash",
                        "name": "bash",
                        "arguments": {"command": "rm missing.txt"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Approval path exercised"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        reviewer = AutoApprovalReviewer(approved=True, reason="test approved")
        config = make_config(tmp_path).model_copy(
            update={"approval_policy": ApprovalPolicy.ON_REQUEST}
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([BashTool()]),
            approval_manager=ApprovalManager(
                reviewer=reviewer,
            ),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "remove file"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert [event.type for event in snapshot.events] == [
            RuntimeEventType.SESSION_STARTED,
            RuntimeEventType.TURN_STARTED,
            RuntimeEventType.USER_MESSAGE,
            RuntimeEventType.MODEL_TOOL_CALL,
            RuntimeEventType.TOOL_CALL_STARTED,
            RuntimeEventType.APPROVAL_REQUESTED,
            RuntimeEventType.APPROVAL_DECIDED,
            RuntimeEventType.TOOL_CALL_FINISHED,
            RuntimeEventType.ASSISTANT_MESSAGE,
            RuntimeEventType.TURN_FINISHED,
        ]
        assert reviewer.requests[0].tool_name == "bash"
        assert snapshot.events[6].payload["approved"] is True
        assert snapshot.events[7].payload["result"]["error"] == "command_failed"

    asyncio.run(run_test())


def test_tool_runner_denies_rejected_workspace_write(tmp_path):
    async def run_test() -> None:
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {"path": "blocked.txt", "content": "nope"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Write was denied"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        reviewer = AutoApprovalReviewer(approved=False, reason="test denied")
        config = make_config(tmp_path).model_copy(
            update={"approval_policy": ApprovalPolicy.ON_REQUEST}
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([WriteFileTool()]),
            approval_manager=ApprovalManager(
                reviewer=reviewer,
            ),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "write blocked"))
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert not (tmp_path / "blocked.txt").exists()
        assert snapshot.events[5].type == RuntimeEventType.APPROVAL_REQUESTED
        assert snapshot.events[6].payload["approved"] is False
        assert snapshot.events[7].payload["result"]["error"] == "approval_denied"
        assert snapshot.events[-1].type == RuntimeEventType.TURN_FINISHED

    asyncio.run(run_test())


def test_thread_approval_decision_allows_pending_tool_call(tmp_path):
    async def run_test() -> None:
        reviewer = ThreadApprovalReviewer()
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {"path": "approved.txt", "content": "yes"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Write approved"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path).model_copy(
            update={"approval_policy": ApprovalPolicy.ON_REQUEST}
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([WriteFileTool()]),
            approval_manager=ApprovalManager(
                reviewer=reviewer,
            ),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "write approved"))
        approval_event = await next_event_of_type(
            thread, RuntimeEventType.APPROVAL_REQUESTED
        )
        assert (
            thread.list_pending_approvals()[0].approval_id
            == approval_event.payload["approval_id"]
        )

        await thread.submit(
            SessionInput.approval_decision(
                "inp_approve",
                approval_id=approval_event.payload["approval_id"],
                approved=True,
                reason="approved in test",
            )
        )
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "yes"
        decided = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.APPROVAL_DECIDED
        ][0]
        assert decided.payload["approved"] is True
        assert decided.payload["reviewer"] == "user"

    asyncio.run(run_test())


def test_thread_approval_decision_denies_pending_tool_call(tmp_path):
    async def run_test() -> None:
        reviewer = ThreadApprovalReviewer()
        provider = MockProvider(
            script=[
                ModelEvent(
                    type=ModelEventType.TOOL_CALL,
                    payload={
                        "call_id": "call_write",
                        "name": "write_file",
                        "arguments": {"path": "denied.txt", "content": "no"},
                    },
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
                ModelEvent(
                    type=ModelEventType.MESSAGE_COMPLETED,
                    payload={"text": "Write denied"},
                ),
                ModelEvent(type=ModelEventType.COMPLETED),
            ]
        )
        config = make_config(tmp_path).model_copy(
            update={"approval_policy": ApprovalPolicy.ON_REQUEST}
        )
        runtime = AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry([provider]),
            tool_registry=ToolRegistry([WriteFileTool()]),
            approval_manager=ApprovalManager(
                reviewer=reviewer,
            ),
        )

        thread = await runtime.create_thread(config)
        await thread.submit(SessionInput.user_message("inp_test", "write denied"))
        approval_event = await next_event_of_type(
            thread, RuntimeEventType.APPROVAL_REQUESTED
        )
        await thread.submit(
            SessionInput.approval_decision(
                "inp_deny",
                approval_id=approval_event.payload["approval_id"],
                approved=False,
                reason="denied in test",
            )
        )
        await thread.wait_until_idle()
        snapshot = await thread.read_snapshot()

        assert not (tmp_path / "denied.txt").exists()
        finished = [
            event
            for event in snapshot.events
            if event.type == RuntimeEventType.TOOL_CALL_FINISHED
        ][0]
        assert finished.payload["result"]["error"] == "approval_denied"

    asyncio.run(run_test())
