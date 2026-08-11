from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from codecraft.approval.policy import ApprovalPolicy
from codecraft.config.settings import RuntimeSettings
from codecraft.sandbox.policy import SandboxMode


_ConfigSource = Literal["user", "profile", "project", "explicit"]
_PROJECT_ALLOWED_FIELDS = {
    "model": frozenset(
        {
            "provider",
            "name",
            "context_window_tokens",
            "max_output_tokens",
        }
    ),
    "instructions": frozenset({"user"}),
    "turn": frozenset(
        {
            "max_tool_calls",
            "max_tool_output_chars",
            "max_tool_output_tokens",
            "turn_timeout_seconds",
            "tool_timeout_seconds",
            "approval_timeout_seconds",
            "context_safety_margin_tokens",
            "context_keep_recent_items",
            "max_parallel_read_tools",
        }
    ),
}


@dataclass(frozen=True)
class _ConfigLayer:
    """一个配置文件路径及其信任来源。"""

    path: Path
    source: _ConfigSource


@dataclass(frozen=True)
class ConfigOverrides:
    """命令行参数转换后的配置覆盖层。"""

    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_cli(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        approval_policy: ApprovalPolicy | None = None,
        sandbox_mode: SandboxMode | None = None,
        network_access: bool | None = None,
        codecraft_home: Path | None = None,
    ) -> ConfigOverrides:
        """把显式 CLI 选项转换成只包含已传值的嵌套覆盖层。

        Args:
            provider: 可选模型 Provider 名称。
            model: 可选模型名称。
            approval_policy: 可选审批策略。
            sandbox_mode: 可选沙箱模式。
            network_access: 可选网络开关；``False`` 仍是显式覆盖值。
            codecraft_home: 可选数据目录。

        Returns:
            可放在所有文件配置之后合并的 ``ConfigOverrides``。

        Example:
            >>> ConfigOverrides.from_cli(
            ...     provider="openai", network_access=False
            ... ).values
            {'model': {'provider': 'openai'}, 'sandbox': {'network_access': False}}
        """
        values: dict[str, Any] = {}
        _set_nested(values, ["model", "provider"], provider)
        _set_nested(values, ["model", "name"], model)
        _set_nested(values, ["approval", "policy"], approval_policy)
        _set_nested(values, ["sandbox", "mode"], sandbox_mode)
        _set_nested(values, ["sandbox", "network_access"], network_access)
        _set_nested(values, ["paths", "codecraft_home"], codecraft_home)
        return cls(values=values)


class ConfigLoader:
    """按默认值、全局配置、profile、项目配置和显式参数加载配置。"""

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        codecraft_home: Path | None = None,
    ) -> None:
        """固定项目目录和用户配置根目录。

        Args:
            cwd: 查找项目 ``.codecraft/config.toml`` 的目录；默认进程 cwd。
            codecraft_home: 用户配置、profiles 的根目录；默认 ``~/.codecraft``。
        """
        self.cwd = (cwd or Path.cwd()).expanduser().resolve()
        self.codecraft_home = (codecraft_home or Path("~/.codecraft")).expanduser()

    def load(
        self,
        *,
        profile: str | None = None,
        config_path: Path | None = None,
        overrides: ConfigOverrides | None = None,
    ) -> RuntimeSettings:
        """按信任和优先级合并配置层，并校验成 ``RuntimeSettings``。

        Args:
            profile: ``codecraft_home/profiles`` 下不含扩展名的 profile 名。
            config_path: 用户显式选择、优先级高于项目配置的 TOML 文件。
            overrides: 最后应用的命令行覆盖层。

        Returns:
            经过全部字段和跨字段约束校验的 Runtime 配置。

        Raises:
            ValueError: 项目配置越权或任意层内容不满足 RuntimeSettings。
            tomllib.TOMLDecodeError: 配置文件不是合法 TOML。
        """
        merged = RuntimeSettings().model_dump(mode="python")
        for layer in self._config_layers(profile=profile, config_path=config_path):
            if not layer.path.exists():
                continue
            values = _read_toml(layer.path)
            if layer.source == "project":
                _validate_project_config(layer.path, values)
            merged = _deep_merge(merged, values)

        if overrides is not None:
            merged = _deep_merge(merged, overrides.values)

        return RuntimeSettings.model_validate(merged)

    def _config_layers(
        self,
        *,
        profile: str | None,
        config_path: Path | None,
    ) -> list[_ConfigLayer]:
        """返回带来源的配置层；后面的层覆盖前面的层。"""
        project_path = self.cwd / ".codecraft" / "config.toml"
        explicit_path = config_path.expanduser() if config_path else None
        layers = [
            _ConfigLayer(self.codecraft_home / "config.toml", "user"),
        ]
        if profile:
            layers.append(
                _ConfigLayer(
                    self.codecraft_home / "profiles" / f"{profile}.toml",
                    "profile",
                )
            )

        if explicit_path is None or explicit_path.resolve() != project_path.resolve():
            layers.append(_ConfigLayer(project_path, "project"))

        if explicit_path is not None:
            layers.append(_ConfigLayer(explicit_path, "explicit"))

        return layers


def _read_toml(path: Path) -> dict[str, Any]:
    """读取一个顶层必须为 table 的 UTF-8 TOML 配置文件。"""
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a TOML table: {path}")
    return data


def _validate_project_config(path: Path, values: dict[str, Any]) -> None:
    """阻止仓库内配置修改审批、沙箱、网络和密钥等安全设置。

    项目文件是不可信仓库内容，只能设置模型非凭据字段、项目指令和 Turn
    预算。用户显式选择的配置不经过此限制，因为该动作本身提供了授权边界。
    """
    restricted: list[str] = []
    for section, section_value in values.items():
        allowed_fields = _PROJECT_ALLOWED_FIELDS.get(section)
        if allowed_fields is None:
            restricted.append(section)
            continue
        if not isinstance(section_value, dict):
            continue
        restricted.extend(
            f"{section}.{field}"
            for field in section_value
            if field not in allowed_fields
        )
    if not restricted:
        return

    fields = ", ".join(sorted(restricted))
    raise ValueError(
        f"untrusted project config {path} may not set security-sensitive "
        f"field(s): {fields}. Move these settings to a user, profile, or "
        "explicit config file."
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典，叶子或容器类型变化时由 override 完全覆盖。

    Example:
        >>> _deep_merge(
        ...     {"model": {"name": "a", "provider": "qwen"}},
        ...     {"model": {"name": "b"}},
        ... )
        {'model': {'name': 'b', 'provider': 'qwen'}}
    """
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result


def _set_nested(values: dict[str, Any], path: list[str], value: Any) -> None:
    """把 CLI 参数写入嵌套配置字典，``None`` 表示不覆盖。

    Example:
        >>> values = {}
        >>> _set_nested(values, ["sandbox", "network_access"], False)
        >>> values
        {'sandbox': {'network_access': False}}
    """
    if value is None:
        return

    target = values
    for key in path[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set nested config value below {key}")
        target = child
    target[path[-1]] = value
