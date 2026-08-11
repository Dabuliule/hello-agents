from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import os
import re
import select
import time
from typing import TypeAlias, cast

from textual.theme import Theme


class TUIThemeMode(StrEnum):
    """用户选择自动探测、强制浅色或强制深色。"""

    AUTO = "auto"
    LIGHT = "light"
    DARK = "dark"


class TUIColorScheme(StrEnum):
    """主题解析后的二值浅/深方案。"""

    LIGHT = "light"
    DARK = "dark"


@dataclass(frozen=True, slots=True)
class TUIColorPalette:
    """Widget/Renderer 共用的语义前景、状态和分隔色。"""

    strong: str
    foreground: str
    secondary: str
    muted: str
    faint: str
    placeholder: str
    separator: str
    accent: str
    success: str
    warning: str
    error: str
    error_detail: str


LIGHT_PALETTE = TUIColorPalette(
    strong="#25282c",
    foreground="#34383a",
    secondary="#5f6770",
    muted="#737b84",
    faint="#7a828b",
    placeholder="#969da5",
    separator="#a3a8ae",
    accent="#416f9f",
    success="#2d7654",
    warning="#936b25",
    error="#b34348",
    error_detail="#9c3f44",
)

DARK_PALETTE = TUIColorPalette(
    strong="#f1f3f5",
    foreground="#d8dbe0",
    secondary="#a5abb3",
    muted="#8b919a",
    faint="#777e87",
    placeholder="#666c74",
    separator="#4f555d",
    accent="#8da2fb",
    success="#79c99e",
    warning="#d8b56d",
    error="#ef767a",
    error_detail="#d98f93",
)


def _theme_variables(
    palette: TUIColorPalette,
    *,
    border: str,
    scrollbar: str,
    selection: str,
    hover: str,
    overlay: str,
    table_header: str,
    cursor_text: str,
) -> dict[str, str]:
    """把 Palette 与 Surface 颜色映射为 Textual/CSS 变量表。"""
    return {
        "codecraft-strong": palette.strong,
        "codecraft-secondary": palette.secondary,
        "codecraft-muted": palette.muted,
        "codecraft-faint": palette.faint,
        "codecraft-placeholder": palette.placeholder,
        "codecraft-separator": palette.separator,
        "codecraft-border": border,
        "codecraft-scrollbar": scrollbar,
        "codecraft-selection": selection,
        "codecraft-selection-text": palette.strong,
        "codecraft-hover": hover,
        "codecraft-overlay": overlay,
        "codecraft-table-header": table_header,
        "codecraft-cursor-text": cursor_text,
        "codecraft-success": palette.success,
        "input-selection-background": selection,
        "input-cursor-background": palette.accent,
        "input-cursor-foreground": cursor_text,
        "scrollbar": scrollbar,
        "scrollbar-background": "transparent",
        "border": border,
        "border-blurred": border,
    }


_LIGHT_VARIABLES = _theme_variables(
    LIGHT_PALETTE,
    border="#cfd3d8",
    scrollbar="#bec4ca",
    selection="#dbe7f5",
    hover="#e2e8f0",
    overlay="#64748b 22%",
    table_header="#e8ebef",
    cursor_text="#ffffff",
)

_DARK_VARIABLES = _theme_variables(
    DARK_PALETTE,
    border="#34383f",
    scrollbar="#3a3f46",
    selection="#293044",
    hover="#343b4e",
    overlay="#000000 62%",
    table_header="#1c1f23",
    cursor_text="#0e0f11",
)

CODECRAFT_LIGHT_THEME = Theme(
    name="codecraft-light",
    primary=LIGHT_PALETTE.accent,
    secondary="#5d7895",
    warning=LIGHT_PALETTE.warning,
    error=LIGHT_PALETTE.error,
    success=LIGHT_PALETTE.success,
    accent=LIGHT_PALETTE.accent,
    foreground=LIGHT_PALETTE.foreground,
    background="#eef0f2",
    surface="#ffffff",
    panel="#f8f9fa",
    dark=False,
    variables=_LIGHT_VARIABLES,
)

CODECRAFT_DARK_THEME = Theme(
    name="codecraft-dark",
    primary=DARK_PALETTE.accent,
    secondary="#7c8ec4",
    warning=DARK_PALETTE.warning,
    error=DARK_PALETTE.error,
    success=DARK_PALETTE.success,
    accent=DARK_PALETTE.accent,
    foreground=DARK_PALETTE.foreground,
    background="#0e0f11",
    surface="#16181b",
    panel="#121417",
    dark=True,
    variables=_DARK_VARIABLES,
)

CODECRAFT_THEMES = (CODECRAFT_LIGHT_THEME, CODECRAFT_DARK_THEME)
CODECRAFT_THEME_VARIABLE_DEFAULTS = _LIGHT_VARIABLES

_OSC_RGB = re.compile(
    rb"\x1b\]11;rgb:([0-9a-f]{1,4})/([0-9a-f]{1,4})/([0-9a-f]{1,4})"
    rb"(?:\x07|\x1b\\)",
    re.IGNORECASE,
)
_OSC_HEX = re.compile(
    rb"\x1b\]11;#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{9}|[0-9a-f]{12})"
    rb"(?:\x07|\x1b\\)",
    re.IGNORECASE,
)
_TerminalAttributes: TypeAlias = list[int | list[bytes | int]]


def palette_for(dark: bool) -> TUIColorPalette:
    """按 Textual current_theme.dark 返回对应不可变 Palette。"""
    return DARK_PALETTE if dark else LIGHT_PALETTE


def textual_theme_name(scheme: TUIColorScheme) -> str:
    """映射 ColorScheme 到已注册的 Textual Theme 名。"""
    return "codecraft-dark" if scheme == TUIColorScheme.DARK else "codecraft-light"


def resolve_color_scheme(
    mode: TUIThemeMode,
    *,
    env: Mapping[str, str] | None = None,
    terminal_background: Callable[[], tuple[int, int, int] | None] | None = None,
    fallback: TUIColorScheme = TUIColorScheme.LIGHT,
) -> TUIColorScheme:
    """解析显式模式，或按 COLORFGBG→OSC 11→fallback 自动选择主题。

    依赖参数可注入以测试；显式 light/dark 不读取环境或终端。COLORFGBG 是
    无 I/O 快路径，只有缺失/无效时才短暂查询终端背景。
    """
    if mode != TUIThemeMode.AUTO:
        return TUIColorScheme(mode.value)

    environment = os.environ if env is None else env
    colorfgbg_scheme = _scheme_from_colorfgbg(environment.get("COLORFGBG"))
    if colorfgbg_scheme is not None:
        return colorfgbg_scheme

    read_background = terminal_background or query_terminal_background
    background = read_background()
    return _scheme_from_rgb(background) if background is not None else fallback


def parse_osc_background(response: bytes) -> tuple[int, int, int] | None:
    """解析 OSC 11 的 rgb: 或 #RGB/#RRGGBB/高位宽响应并缩放到 8-bit。"""
    rgb_match = _OSC_RGB.search(response)
    if rgb_match is not None:
        red, green, blue = rgb_match.groups()
        return (
            _scale_hex_channel(red),
            _scale_hex_channel(green),
            _scale_hex_channel(blue),
        )

    hex_match = _OSC_HEX.search(response)
    if hex_match is None:
        return None
    value = hex_match.group(1)
    width = len(value) // 3
    return (
        _scale_hex_channel(value[:width]),
        _scale_hex_channel(value[width : width * 2]),
        _scale_hex_channel(value[width * 2 :]),
    )


def query_terminal_background(timeout: float = 0.1) -> tuple[int, int, int] | None:
    """通过 OSC 11 查询终端背景色；不支持该协议时快速返回。"""
    try:
        import termios
    except ImportError:
        return None

    flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        terminal_fd = os.open("/dev/tty", flags)
    except OSError:
        return None

    original_attributes: _TerminalAttributes | None = None
    try:
        if select.select([terminal_fd], [], [], 0)[0]:
            return None

        original_attributes = cast(
            _TerminalAttributes,
            termios.tcgetattr(terminal_fd),
        )
        query_attributes = deepcopy(original_attributes)
        query_attributes[3] = cast(int, query_attributes[3]) & ~(
            termios.ICANON | termios.ECHO
        )
        control_characters = cast(list[bytes | int], query_attributes[6])
        control_characters[termios.VMIN] = 0
        control_characters[termios.VTIME] = 0
        termios.tcsetattr(terminal_fd, termios.TCSANOW, query_attributes)

        os.write(terminal_fd, b"\x1b]11;?\x1b\\")
        deadline = time.monotonic() + max(timeout, 0)
        response = bytearray()
        while len(response) < 256:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([terminal_fd], [], [], remaining)[0]:
                break
            response.extend(os.read(terminal_fd, 256 - len(response)))
            background = parse_osc_background(bytes(response))
            if background is not None:
                return background
        return None
    except (OSError, ValueError):
        return None
    finally:
        if original_attributes is not None:
            try:
                termios.tcsetattr(
                    terminal_fd,
                    termios.TCSANOW,
                    original_attributes,
                )
            except OSError:
                pass
        os.close(terminal_fd)


def _scheme_from_colorfgbg(value: str | None) -> TUIColorScheme | None:
    """解析 COLORFGBG 最后 ANSI 0-15 背景索引并判断浅/深。"""
    if not value:
        return None
    try:
        background_index = int(re.split(r"[;:]", value)[-1])
    except ValueError:
        return None
    if background_index not in range(16):
        return None
    return (
        TUIColorScheme.LIGHT
        if background_index in {7, 9, 10, 11, 12, 13, 14, 15}
        else TUIColorScheme.DARK
    )


def _scheme_from_rgb(rgb: tuple[int, int, int]) -> TUIColorScheme:
    """用加权感知亮度阈值 0.5 把 RGB 分类为浅/深。"""
    red, green, blue = rgb
    brightness = (299 * red + 587 * green + 114 * blue) / 255_000
    return TUIColorScheme.LIGHT if brightness >= 0.5 else TUIColorScheme.DARK


def _scale_hex_channel(value: bytes) -> int:
    """把 1-4 位十六进制通道按其最大值线性缩放到 0-255。"""
    maximum = (16 ** len(value)) - 1
    return int(round(int(value, 16) * 255 / maximum))
