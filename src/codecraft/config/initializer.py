"""首次 CLI 启动时创建可见、可编辑的用户配置。

初始化只发生在 CLI bootstrap 边界。``ConfigLoader`` 本身保持只读，因此测试、
嵌入调用和单纯解析配置不会意外修改用户目录。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from codecraft.config.provider_defaults import (
    default_api_key_env,
    default_base_url,
)
from codecraft.config.settings import RuntimeSettings


def ensure_user_config(codecraft_home: Path) -> Path:
    """在用户配置缺失时以排他方式创建 ``config.toml``。

    Args:
        codecraft_home: 用户配置、Session、Skill 和索引共用的数据根目录。

    Returns:
        展开并规范化后的用户配置文件路径。

    多个 CodeCraft 进程可能同时首次启动，所以不能采用“先判断不存在，再普通
    写入”的竞态写法。函数先完整写入同目录临时文件，再用 ``os.link`` 排他发布：
    只有一个进程能创建目标硬链接，其他进程把 ``FileExistsError`` 当作初始化
    已完成。目标路径要么不存在，要么指向完整文件；已有配置不会被覆盖。
    """
    home = codecraft_home.expanduser().resolve()
    config_path = home / "config.toml"
    home.mkdir(parents=True, exist_ok=True)

    # 这个检查只是避免每次启动都创建临时文件。它不承担并发正确性；检查后仍
    # 可能有另一个进程抢先发布，最终由 os.link 的排他语义裁决。
    if config_path.exists():
        return config_path

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".config.toml.",
            suffix=".tmp",
            dir=home,
            delete=False,
        ) as config_file:
            temporary_path = Path(config_file.name)
            config_file.write(render_default_user_config())
            config_file.flush()
            os.fsync(config_file.fileno())

        try:
            # link 不会替换已存在目标；成功后 config.toml 与已完整写入的临时文件
            # 指向同一 inode，所以读取者不会观察到半份 TOML。
            os.link(temporary_path, config_path)
        except FileExistsError:
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return config_path


def render_default_user_config(settings: RuntimeSettings | None = None) -> str:
    """把当前内置 Runtime 默认值渲染为带说明的用户级 TOML。

    数值、权限和模型选择来自 ``RuntimeSettings``；Provider 连接兜底来自与实际
    Provider 共用的 ``provider_defaults``。因此修改运行时默认值后，测试能直接
    发现生成文件是否漂移。模板只写环境变量名，绝不把环境变量值写入磁盘。

    Args:
        settings: 可选设置对象，主要用于测试；缺失时使用当前内置默认值。

    Returns:
        末尾带换行、可直接由 ``tomllib`` 解析的 TOML 文本。
    """
    defaults = settings or RuntimeSettings()
    provider = defaults.model.provider
    api_key_env = defaults.model.api_key_env or default_api_key_env(provider)
    base_url = defaults.model.base_url or default_base_url(provider)
    docker = defaults.sandbox.docker

    model_lines = [
        f"provider = {_toml_string(provider)}",
        f"name = {_toml_string(defaults.model.name)}",
    ]
    if api_key_env is not None:
        model_lines.append(f"api_key_env = {_toml_string(api_key_env)}")
    if base_url is not None:
        model_lines.append(f"base_url = {_toml_string(base_url)}")
    model_lines.extend(
        [
            f"context_window_tokens = {defaults.model.context_window_tokens}",
            f"max_output_tokens = {defaults.model.max_output_tokens}",
        ]
    )

    lines = [
        "# CodeCraft generated this file on first startup.",
        "# Existing config files are never overwritten automatically.",
        "# Store the API key in the named environment variable, not in this file.",
        "",
        "[model]",
        *model_lines,
        "",
        "[approval]",
        f"policy = {_toml_string(defaults.approval.policy.value)}",
        "",
        "[sandbox]",
        f"mode = {_toml_string(defaults.sandbox.mode.value)}",
        f"backend = {_toml_string(defaults.sandbox.backend.value)}",
        f"network_access = {_toml_bool(defaults.sandbox.network_access)}",
        f"env_allowlist = {_toml_string_list(defaults.sandbox.env_allowlist)}",
        "",
        "[sandbox.docker]",
        f"image = {_toml_string(docker.image)}",
        f"cpus = {docker.cpus}",
        f"memory_mb = {docker.memory_mb}",
        f"pids_limit = {docker.pids_limit}",
        f"tmpfs_mb = {docker.tmpfs_mb}",
        "",
        "[instructions]",
        '# user = "Add your persistent instructions here."',
        "",
        "[turn]",
        f"max_tool_calls = {defaults.turn.max_tool_calls}",
        f"max_tool_output_chars = {defaults.turn.max_tool_output_chars}",
        f"max_tool_output_tokens = {defaults.turn.max_tool_output_tokens}",
        f"turn_timeout_seconds = {defaults.turn.turn_timeout_seconds}",
        f"tool_timeout_seconds = {defaults.turn.tool_timeout_seconds}",
        f"approval_timeout_seconds = {defaults.turn.approval_timeout_seconds}",
        (
            "context_safety_margin_tokens = "
            f"{defaults.turn.context_safety_margin_tokens}"
        ),
        f"context_keep_recent_items = {defaults.turn.context_keep_recent_items}",
        f"max_parallel_read_tools = {defaults.turn.max_parallel_read_tools}",
        "",
        "[mcp]",
        "# Add trusted stdio servers under [mcp.servers.<name>].",
        "",
    ]
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    """使用 JSON 的兼容转义规则生成 TOML basic string。"""
    return json.dumps(value, ensure_ascii=False)


def _toml_bool(value: bool) -> str:
    """按 TOML 语法输出小写布尔值。"""
    return "true" if value else "false"


def _toml_string_list(values: list[str]) -> str:
    """把环境变量名称列表渲染为 TOML array。"""
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"
