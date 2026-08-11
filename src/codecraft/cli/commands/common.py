from __future__ import annotations

from codecraft.cli.ui import RenderConfig, RuntimeEventRenderer, make_console
from codecraft.core.errors import CodecraftError


def build_event_renderer(*, debug: bool = False) -> RuntimeEventRenderer:
    """用标准 Rich Console 和 debug 开关构造 CLI 事件渲染器。"""
    console = make_console()
    return RuntimeEventRenderer(
        console=console,
        render_config=RenderConfig(debug=debug),
    )


def render_startup_error(error: CodecraftError) -> None:
    """向 stderr 输出稳定 CodecraftError 消息/码和可选建议。"""
    console = make_console(stderr=True)
    console.print(f"{error.message} ({error.code})", style="error", markup=False)
    if error.suggestion:
        console.print(error.suggestion, style="muted", markup=False, soft_wrap=True)
