from __future__ import annotations

import asyncio
import json

from typer.testing import CliRunner

from codecraft.approval.manager import ApprovalManager
from codecraft.approval.thread_reviewer import ThreadApprovalReviewer
from codecraft.cli import app as cli_app
from codecraft.cli.app import app
from codecraft.core.runtime import AgentRuntime
from codecraft.core.ids import new_id
from codecraft.core.session_store import SessionStore
from codecraft.llm import LLMProviderRegistry, MockProvider, ModelEvent, ModelEventType
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSource
from codecraft.tool import BashTool, ToolRegistry


runner = CliRunner()


def make_config(tmp_path) -> SessionConfig:
    return SessionConfig(
        session_id="ses_cli",
        source=SessionSource.TEST,
        cwd=tmp_path,
        codecraft_home=tmp_path / ".codecraft",
        model="mock-model",
        model_provider="mock",
        approval_policy="never",
        sandbox_mode="workspace_write",
    )


def seed_session(tmp_path) -> SessionConfig:
    async def run() -> SessionConfig:
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
                turn_id="turn_cli",
                seq=2,
                type=RuntimeEventType.TURN_FINISHED,
                payload={"answer": "done", "tool_calls": 0, "duration_ms": 1},
            )
        )
        return config

    return asyncio.run(run())


def test_cli_uses_root_as_the_only_interactive_entrypoint():
    command_names = {command.name for command in app.registered_commands}

    assert "tui" not in command_names
    assert "exec" in command_names
    assert "chat" not in command_names
    assert "resume" not in command_names

    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "--theme" in help_result.output
    assert "auto" in help_result.output


def test_sessions_command_lists_session(tmp_path):
    config = seed_session(tmp_path)

    result = runner.invoke(
        app,
        ["sessions", "--codecraft-home", str(config.codecraft_home)],
    )

    assert result.exit_code == 0
    assert "Recent Sessions" in result.output
    assert "ses_cli" in result.output
    assert "valid" in result.output


def test_sessions_all_marks_invalid_session(tmp_path):
    async def seed() -> None:
        config = make_config(tmp_path)
        bad_config = config.model_copy(update={"session_id": "ses_bad"})
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

    asyncio.run(seed())

    default_result = runner.invoke(
        app,
        ["sessions", "--codecraft-home", str(tmp_path / ".codecraft")],
    )
    all_result = runner.invoke(
        app,
        ["sessions", "--codecraft-home", str(tmp_path / ".codecraft"), "--all"],
    )

    assert default_result.exit_code == 0
    assert "ses_cli" in default_result.output
    assert "ses_bad" not in default_result.output
    assert all_result.exit_code == 0
    assert "ses_cli" in all_result.output
    assert "ses_bad" in all_result.output
    assert "valid" in all_result.output
    assert "invalid:" in all_result.output


def test_inspect_command_prints_summary_and_events(tmp_path):
    config = seed_session(tmp_path)

    result = runner.invoke(
        app,
        [
            "inspect",
            config.session_id,
            "--codecraft-home",
            str(config.codecraft_home),
            "--events",
        ],
    )

    assert result.exit_code == 0
    assert "session inspect" in result.output
    assert "ses_cli" in result.output
    assert "Events" in result.output
    assert "session_started" in result.output
    assert "turn_finished" in result.output
    assert "done" in result.output


def test_inspect_raw_prints_invalid_session_lines(tmp_path):
    async def seed() -> SessionConfig:
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
        return config

    config = asyncio.run(seed())

    raw_result = runner.invoke(
        app,
        [
            "inspect",
            config.session_id,
            "--codecraft-home",
            str(config.codecraft_home),
            "--raw",
        ],
    )

    assert raw_result.exit_code == 0
    assert "session_id: ses_cli" in raw_result.output
    assert "raw_lines: 1" in raw_result.output
    assert '1: {"schema_version":1,' in raw_result.output
    assert '"event_id":"evt_' in raw_result.output
    assert '"seq":2' in raw_result.output


def test_inspect_missing_session_prints_friendly_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "inspect",
            "missing",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 1
    assert "No session found: missing" in result.output
    assert "Traceback" not in result.output


def test_trace_command_writes_json_and_html_reports(tmp_path):
    async def seed() -> SessionConfig:
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
                turn_id="turn_trace",
                seq=2,
                type=RuntimeEventType.USER_MESSAGE,
                payload={"input_id": "inp_trace", "text": "read README"},
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_trace",
                seq=3,
                type=RuntimeEventType.MODEL_TOOL_CALL,
                payload={
                    "call_id": "call_read",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_trace",
                seq=4,
                type=RuntimeEventType.TOOL_CALL_STARTED,
                payload={
                    "call_id": "call_read",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_trace",
                seq=5,
                type=RuntimeEventType.TOOL_CALL_FINISHED,
                payload={
                    "call_id": "call_read",
                    "name": "read_file",
                    "result": {
                        "success": True,
                        "content": "README content",
                        "metadata": {},
                    },
                    "duration_ms": 7,
                },
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_trace",
                seq=6,
                type=RuntimeEventType.TURN_FINISHED,
                payload={"answer": "done", "tool_calls": 1, "duration_ms": 12},
            )
        )
        return config

    config = asyncio.run(seed())
    output_dir = tmp_path / "traces"

    result = runner.invoke(
        app,
        [
            "trace",
            config.session_id,
            "--codecraft-home",
            str(config.codecraft_home),
            "--output-dir",
            str(output_dir),
        ],
    )

    json_path = output_dir / "ses_cli.trace.json"
    html_path = output_dir / "ses_cli.trace.html"
    assert result.exit_code == 0
    assert f"trace_json: {json_path}" in result.output
    assert f"trace_html: {html_path}" in result.output
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["session"]["session_id"] == "ses_cli"
    assert report["metrics"]["event_count"] == 6
    assert report["metrics"]["tool_call_count"] == 1
    assert report["metrics"]["tool_failure_count"] == 0
    assert report["turns"][0]["turn_id"] == "turn_trace"
    assert report["tool_calls"][0]["name"] == "read_file"
    assert report["tool_calls"][0]["arguments"] == {"path": "README.md"}
    html = html_path.read_text(encoding="utf-8")
    assert "CodeCraft Trace ses_cli" in html
    assert "read_file" in html


def test_trace_missing_session_prints_friendly_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "trace",
            "missing",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 1
    assert "No session found: missing" in result.output
    assert "Traceback" not in result.output


def test_inspect_command_prints_tool_and_error_summaries(tmp_path):
    async def seed() -> SessionConfig:
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
                turn_id="turn_cli",
                seq=2,
                type=RuntimeEventType.MODEL_TOOL_CALL,
                payload={
                    "call_id": "call_read",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                },
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_cli",
                seq=3,
                type=RuntimeEventType.TOOL_CALL_FINISHED,
                payload={
                    "call_id": "call_read",
                    "name": "read_file",
                    "result": {
                        "success": False,
                        "content": "File does not exist.",
                        "error": "file_not_found",
                        "metadata": {},
                    },
                    "duration_ms": 4,
                },
            )
        )
        await store.append_event(
            RuntimeEvent(
                event_id=new_id("evt_"),
                session_id=config.session_id,
                turn_id="turn_cli",
                seq=4,
                type=RuntimeEventType.ERROR,
                payload={
                    "code": "runtime_error",
                    "message": "runtime failed",
                    "metadata": {},
                },
            )
        )
        return config

    config = asyncio.run(seed())

    result = runner.invoke(
        app,
        [
            "inspect",
            config.session_id,
            "--codecraft-home",
            str(config.codecraft_home),
            "--tools",
            "--errors",
        ],
    )

    assert result.exit_code == 0
    assert "Tool Events" in result.output
    assert "read_file" in result.output
    assert "requested" in result.output
    assert "failed" in result.output
    assert "File does not exist." in result.output
    assert "Errors" in result.output
    assert "tool_error" in result.output
    assert "runtime failed" in result.output


def test_exec_mcp_startup_failure_prints_friendly_error(tmp_path):
    config_path = tmp_path / "mcp.toml"
    config_path.write_text(
        """
[mcp.servers.missing]
command = "codecraft-command-that-does-not-exist"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "exec",
            "test mcp",
            "--config",
            str(config_path),
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 1
    assert "Could not start MCP server 'missing'." in result.output
    assert "mcp_connection_failed" in result.output
    assert "Traceback" not in result.output


def test_exec_command_runs_runtime_and_prints_answer(tmp_path, monkeypatch):
    seen_configs: list[SessionConfig] = []

    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        seen_configs.append(config)
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.MESSAGE_COMPLETED,
                                payload={"text": "exec answer"},
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "answer",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert "exec answer" in result.output
    assert seen_configs[0].model_provider == "mock"
    assert seen_configs[0].model == "mock-model"

    sessions_result = runner.invoke(
        app,
        ["sessions", "--codecraft-home", str(tmp_path / ".codecraft")],
    )
    assert sessions_result.exit_code == 0
    assert "Recent Sessions" in sessions_result.output
    assert "5" in sessions_result.output


def test_exec_command_renders_markdown_assistant_message(tmp_path, monkeypatch):
    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.MESSAGE_COMPLETED,
                                payload={
                                    "text": (
                                        "这是一个为 **AI Agent / Agent Engineer / "
                                        "大模型应用开发岗位** 准备的 **LaTeX 简历模板**项目。"
                                    ),
                                },
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "answer",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert "AI Agent / Agent Engineer / 大模型应用开发岗位" in result.output
    assert "**AI Agent" not in result.output
    assert "**LaTeX" not in result.output


def test_exec_command_does_not_duplicate_streamed_assistant_message(
    tmp_path, monkeypatch
):
    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.MESSAGE_DELTA,
                                payload={"text": "hello "},
                            ),
                            ModelEvent(
                                type=ModelEventType.MESSAGE_DELTA,
                                payload={"text": "stream"},
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "answer",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert result.output.count("hello stream") == 1


def test_exec_command_renders_streamed_markdown_assistant_message(
    tmp_path, monkeypatch
):
    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.MESSAGE_DELTA,
                                payload={"text": "### 5. 当前限制 (v1.0 阶段)\n"},
                            ),
                            ModelEvent(
                                type=ModelEventType.MESSAGE_DELTA,
                                payload={
                                    "text": (
                                        "*   没有操作系统级别的沙箱隔离。\n\n"
                                        "**一句话概括**：CodeCraft 是一个注重"
                                        "**安全性、可配置性和会话可追溯性**的本地 AI 编程助手。"
                                    ),
                                },
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "answer",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert "当前限制 (v1.0 阶段)" in result.output
    assert "一句话概括" in result.output
    assert "### 5." not in result.output
    assert "**一句话概括**" not in result.output
    assert "**安全性" not in result.output


def test_exec_command_loads_config_file_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
provider = "mock"
name = "configured-model"
api_key_env = "MOCK_API_KEY"
base_url = "https://example.test/v1"
context_window_tokens = 65536
max_output_tokens = 4096

[instructions]
user = "Be terse."

[turn]
max_tool_calls = 7
max_tool_output_chars = 2048
max_tool_output_tokens = 1024
turn_timeout_seconds = 600
tool_timeout_seconds = 90
approval_timeout_seconds = 45
context_safety_margin_tokens = 512
context_keep_recent_items = 6
max_parallel_read_tools = 2
""",
        encoding="utf-8",
    )
    seen_configs: list[SessionConfig] = []

    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        seen_configs.append(config)
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.MESSAGE_COMPLETED,
                                payload={"text": "configured answer"},
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry(),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "answer",
            "--config",
            str(config_path),
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
    )

    assert result.exit_code == 0
    assert "configured answer" in result.output
    assert seen_configs[0].model_provider == "mock"
    assert seen_configs[0].model == "configured-model"
    assert seen_configs[0].model_api_key_env == "MOCK_API_KEY"
    assert seen_configs[0].model_base_url == "https://example.test/v1"
    assert seen_configs[0].model_context_window_tokens == 65_536
    assert seen_configs[0].model_max_output_tokens == 4096
    assert seen_configs[0].user_instructions == "Be terse."
    assert seen_configs[0].max_tool_calls == 7
    assert seen_configs[0].max_tool_output_chars == 2048
    assert seen_configs[0].max_tool_output_tokens == 1024
    assert seen_configs[0].turn_timeout_seconds == 600
    assert seen_configs[0].tool_timeout_seconds == 90
    assert seen_configs[0].approval_timeout_seconds == 45
    assert seen_configs[0].context_safety_margin_tokens == 512
    assert seen_configs[0].context_keep_recent_items == 6
    assert seen_configs[0].max_parallel_read_tools == 2


def test_provider_registry_receives_session_model_connection_config(tmp_path):
    config = make_config(tmp_path).model_copy(
        update={
            "model_provider": "qwen",
            "model_api_key_env": "CUSTOM_API_KEY",
            "model_base_url": "https://example.test/v1",
        }
    )

    registry = cli_app._build_provider_registry(config)
    openai = registry.get("openai")
    qwen = registry.get("qwen")
    deepseek = registry.get("deepseek")

    assert openai.api_key_env == "OPENAI_API_KEY"
    assert openai.base_url == "https://example.test/v1"
    assert qwen.api_key_env == "CUSTOM_API_KEY"
    assert qwen.base_url == "https://example.test/v1"
    assert deepseek.api_key_env == "DEEPSEEK_API_KEY"
    assert deepseek.base_url == "https://example.test/v1"


def test_model_api_key_env_uses_provider_defaults_when_unconfigured():
    assert cli_app._model_api_key_env("qwen", None) == "DASHSCOPE_API_KEY"
    assert cli_app._model_api_key_env("openai", None) == "OPENAI_API_KEY"
    assert cli_app._model_api_key_env("deepseek", None) == "DEEPSEEK_API_KEY"
    assert cli_app._model_api_key_env("mock", None) is None
    assert cli_app._model_api_key_env("qwen", "CUSTOM_API_KEY") == "CUSTOM_API_KEY"


def test_exec_command_prints_bash_approval_details(tmp_path, monkeypatch):
    def fake_runtime(config: SessionConfig) -> AgentRuntime:
        return AgentRuntime(
            session_store=SessionStore(config.codecraft_home),
            llm_providers=LLMProviderRegistry(
                [
                    MockProvider(
                        [
                            ModelEvent(
                                type=ModelEventType.TOOL_CALL,
                                payload={
                                    "call_id": "call_bash",
                                    "name": "bash",
                                    "arguments": {"command": "python -c 'print(1)'"},
                                },
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                            ModelEvent(
                                type=ModelEventType.MESSAGE_COMPLETED,
                                payload={"text": "Command was not run."},
                            ),
                            ModelEvent(type=ModelEventType.COMPLETED),
                        ]
                    )
                ]
            ),
            tool_registry=ToolRegistry([BashTool()]),
            approval_manager=ApprovalManager(
                reviewer=ThreadApprovalReviewer(),
            ),
        )

    monkeypatch.setattr(cli_app, "_build_runtime", fake_runtime)

    result = runner.invoke(
        app,
        [
            "exec",
            "check python",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--approval-policy",
            "on_request",
            "--codecraft-home",
            str(tmp_path / ".codecraft"),
        ],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "• bash python -c 'print(1)'" in result.output
    assert "approval required" in result.output
    assert "python -c 'print(1)'" in result.output
    assert "Approve?" in result.output
    assert "approval rejected" in result.output
    assert "✗ bash failed" in result.output
