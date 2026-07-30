from __future__ import annotations

from codecraft.config import ConfigLoader, ConfigOverrides


def test_config_loader_applies_precedence_order(tmp_path):
    home = tmp_path / "home"
    cwd = tmp_path / "workspace"
    home.mkdir()
    cwd.mkdir()
    (home / "profiles").mkdir()
    (cwd / ".codecraft").mkdir()

    (home / "config.toml").write_text(
        """
[model]
provider = "openai"
name = "gpt-user"

[approval]
policy = "never"

[sandbox]
network_access = true
""",
        encoding="utf-8",
    )
    (home / "profiles" / "fast.toml").write_text(
        """
[model]
name = "qwen-fast"
""",
        encoding="utf-8",
    )
    (cwd / ".codecraft" / "config.toml").write_text(
        """
[model]
provider = "qwen"
context_window_tokens = 65536
max_output_tokens = 4096

[instructions]
user = "Always answer in Chinese."

[turn]
max_tool_calls = 12
max_tool_output_chars = 4096
max_tool_output_tokens = 1024
turn_timeout_seconds = 900
tool_timeout_seconds = 120
approval_timeout_seconds = 60
context_safety_margin_tokens = 512
context_keep_recent_items = 8
max_parallel_read_tools = 3
""",
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(
        """
[approval]
policy = "untrusted"
""",
        encoding="utf-8",
    )

    settings = ConfigLoader(cwd=cwd, codecraft_home=home).load(
        profile="fast",
        config_path=explicit,
        overrides=ConfigOverrides.from_cli(model="qwen-cli"),
    )

    assert settings.model.provider == "qwen"
    assert settings.model.name == "qwen-cli"
    assert settings.model.context_window_tokens == 65_536
    assert settings.model.max_output_tokens == 4096
    assert settings.approval.policy == "untrusted"
    assert settings.sandbox.network_access is True
    assert settings.instructions.user == "Always answer in Chinese."
    assert settings.turn.max_tool_calls == 12
    assert settings.turn.max_tool_output_chars == 4096
    assert settings.turn.max_tool_output_tokens == 1024
    assert settings.turn.turn_timeout_seconds == 900
    assert settings.turn.tool_timeout_seconds == 120
    assert settings.turn.approval_timeout_seconds == 60
    assert settings.turn.context_safety_margin_tokens == 512
    assert settings.turn.context_keep_recent_items == 8
    assert settings.turn.max_parallel_read_tools == 3


def test_config_loader_uses_builtin_defaults_when_files_are_missing(tmp_path):
    settings = ConfigLoader(cwd=tmp_path, codecraft_home=tmp_path / "home").load()

    assert settings.model.provider == "qwen"
    assert settings.model.name == "qwen-plus"
    assert settings.model.api_key_env is None
    assert settings.model.context_window_tokens == 131_072
    assert settings.model.max_output_tokens == 8192
    assert settings.approval.policy == "on_request"
    assert settings.sandbox.mode == "workspace_write"
    assert settings.sandbox.network_access is False
    assert settings.instructions.user is None
    assert settings.turn.max_tool_calls == 30
    assert settings.turn.max_tool_output_chars == 80_000
    assert settings.turn.max_tool_output_tokens == 16_384
    assert settings.turn.turn_timeout_seconds == 1800
    assert settings.turn.tool_timeout_seconds == 300
    assert settings.turn.approval_timeout_seconds == 300
    assert settings.turn.context_safety_margin_tokens == 2048
    assert settings.turn.context_keep_recent_items == 12
    assert settings.turn.max_parallel_read_tools == 4
