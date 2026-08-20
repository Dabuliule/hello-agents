from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import tomllib

from codecraft.cli.bootstrap import load_session_config
from codecraft.config import ensure_user_config, render_default_user_config
from codecraft.config.provider_defaults import QWEN_BASE_URL
from codecraft.schema.session import SessionSource


def test_ensure_user_config_creates_parseable_current_defaults(tmp_path, monkeypatch):
    home = tmp_path / "nested" / ".codecraft"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "must-not-be-persisted")

    config_path = ensure_user_config(home)
    text = config_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(text)

    assert config_path == home.resolve() / "config.toml"
    assert parsed["model"] == {
        "provider": "qwen",
        "name": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": QWEN_BASE_URL,
        "context_window_tokens": 131_072,
        "max_output_tokens": 8192,
    }
    assert parsed["approval"]["policy"] == "on_request"
    assert parsed["sandbox"]["mode"] == "workspace_write"
    assert parsed["sandbox"]["network_access"] is False
    assert parsed["sandbox"]["docker"]["image"] == "codecraft-sandbox:py311"
    assert parsed["turn"]["max_tool_calls"] == 30
    assert "must-not-be-persisted" not in text


def test_ensure_user_config_never_overwrites_existing_file(tmp_path):
    home = tmp_path / ".codecraft"
    home.mkdir()
    config_path = home / "config.toml"
    original = b'# user-owned bytes\n[model]\nname = "custom"\n'
    config_path.write_bytes(original)

    returned_path = ensure_user_config(home)

    assert returned_path == config_path
    assert config_path.read_bytes() == original


def test_concurrent_first_startup_publishes_one_complete_config(tmp_path):
    home = tmp_path / ".codecraft"
    workers = 8
    barrier = Barrier(workers)

    def initialize_after_barrier() -> Path:
        barrier.wait()
        return ensure_user_config(home)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        paths = list(executor.map(lambda _: initialize_after_barrier(), range(workers)))

    config_path = home / "config.toml"
    assert paths == [config_path] * workers
    assert config_path.read_text(encoding="utf-8") == render_default_user_config()
    assert list(home.glob(".config.toml.*.tmp")) == []
    assert (
        tomllib.loads(config_path.read_text(encoding="utf-8"))["model"]["provider"]
        == "qwen"
    )


def test_session_bootstrap_creates_and_immediately_uses_user_config(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / ".codecraft"
    monkeypatch.chdir(workspace)

    config = load_session_config(
        source=SessionSource.TEST,
        provider=None,
        model=None,
        codecraft_home=home,
        config_path=None,
        profile=None,
        approval_policy=None,
        network=None,
    )

    assert (home / "config.toml").is_file()
    assert config.codecraft_home == home.resolve()
    assert config.model_provider == "qwen"
    assert config.model == "qwen-plus"
    assert config.model_api_key_env == "DASHSCOPE_API_KEY"
    assert config.model_base_url == QWEN_BASE_URL
