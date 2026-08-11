from __future__ import annotations

from typing import Any

from rich.console import Console

from codecraft.cli.ui.render_config import RenderConfig
from codecraft.schema.event import EventPayload


class ToolRenderer:
    """按内置 Tool 语义渲染开始/成功/失败摘要和输出预览。"""

    def __init__(self, console: Console, config: RenderConfig) -> None:
        """绑定 Console/config，并初始化 call_id 到开始参数的临时映射。"""
        self.console = console
        self.config = config
        self._started_args: dict[str, dict[str, Any]] = {}

    def render_started(self, payload: EventPayload) -> None:
        """保存合法 arguments 供完成事件补路径，并打印开始摘要。"""
        name = str(payload.get("name") or "tool")
        arguments = payload.get("arguments")
        if isinstance(arguments, dict):
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                self._started_args[call_id] = arguments
        self.console.print(
            self._format_started(name, arguments), markup=False, soft_wrap=True
        )

    def render_finished(self, payload: EventPayload) -> None:
        """按 result.success 渲染语义统计；失败另输出头尾保留的内容预览。"""
        name = str(payload.get("name") or "tool")
        result = payload.get("result")
        duration_ms = payload.get("duration_ms")
        if not isinstance(result, dict):
            self.console.print(f"✓ {name} completed", markup=False, soft_wrap=True)
            return

        success = result.get("success") is True
        call_id = payload.get("call_id")
        arguments = (
            self._started_args.pop(call_id, {}) if isinstance(call_id, str) else {}
        )
        if success:
            self.console.print(
                self._format_success(name, result, duration_ms, arguments),
                markup=False,
                soft_wrap=True,
            )
        else:
            content = preview_tool_output(
                str(result.get("content") or result.get("error") or ""),
                self.config.max_tool_preview_chars,
            )
            self.console.print(
                self._format_failure(name, duration_ms, content),
                markup=False,
                soft_wrap=True,
            )
            self.console.print(content)

    def render_patch_applied(self, payload: EventPayload) -> None:
        """显示 Patch 附加事件中的修改/新增/删除数量。"""
        self.console.print(
            f"✓ patch applied · {payload.get('modified', 0)} modified · "
            f"{payload.get('added', 0)} added · {payload.get('deleted', 0)} deleted",
            markup=False,
            soft_wrap=True,
        )

    def _format_started(self, name: str, arguments: Any) -> str:
        """为搜索/Bash 提取关键参数，其余 Tool 输出紧凑名称。"""
        if name == "read_file":
            return "• read_file"
        if name == "list_files":
            return "• list_files"
        if name == "workspace_search":
            query = ""
            if isinstance(arguments, dict):
                query = str(arguments.get("query") or "")
            return f"• workspace_search {query}" if query else "• workspace_search"
        if name == "write_file":
            return "• write_file"
        if name == "apply_patch":
            return "• apply_patch"
        if name == "bash":
            command = ""
            if isinstance(arguments, dict):
                command = str(arguments.get("command") or "")
            return f"• bash {command}" if command else "• bash"
        return f"• {name}"

    def _format_success(
        self,
        name: str,
        result: dict[str, Any],
        duration_ms: Any,
        arguments: dict[str, Any],
    ) -> str:
        """按 Tool 类型提取路径、行数、字节、命中、策略、状态和耗时。"""
        duration = f" · {duration_ms}ms" if isinstance(duration_ms, int) else ""
        if name == "read_file":
            path = self._display_path(result, arguments)
            lines = self._line_count(result)
            size = self._byte_size(result)
            parts = [f"✓ read_file {path}"]
            if lines is not None:
                parts.append(f"{lines} lines")
            if size is not None:
                parts.append(format_bytes(size))
            return " · ".join(parts)
        if name == "list_files":
            path = self._display_path(result, arguments)
            count = result.get("metadata", {}).get("count")
            suffix = f" · {count} entries" if isinstance(count, int) else ""
            return f"✓ list_files {path}{suffix}{duration}"
        if name == "workspace_search":
            metadata = result.get("metadata", {})
            query = metadata.get("query")
            count = metadata.get("match_count")
            retriever = metadata.get("retriever")
            fallback_from = metadata.get("fallback_from")
            query_text = f" {query}" if isinstance(query, str) and query else ""
            suffix = f" · {count} matches" if isinstance(count, int) else ""
            if isinstance(retriever, str) and retriever:
                suffix += f" · {retriever}"
            if isinstance(fallback_from, str) and fallback_from:
                suffix += f" fallback from {fallback_from}"
            return f"✓ workspace_search{query_text}{suffix}{duration}"
        if name == "write_file":
            metadata = result.get("metadata", {})
            path = self._display_path(result, arguments)
            status = metadata.get("status") or "wrote"
            size = metadata.get("bytes")
            suffix = f" · {format_bytes(size)}" if isinstance(size, int) else ""
            return f"✓ write_file {status} {path}{suffix}{duration}"
        if name == "apply_patch":
            return f"✓ apply_patch completed{duration}"
        return f"✓ {name} completed{duration}"

    def _format_failure(self, name: str, duration_ms: Any, content: str) -> str:
        """组合失败 Tool 名、可选耗时和紧凑原因。"""
        duration = f" · {duration_ms}ms" if isinstance(duration_ms, int) else ""
        summary = f": {content}" if content else ""
        return f"✗ {name} failed{duration}{summary}"

    def _display_path(self, result: dict[str, Any], arguments: dict[str, Any]) -> str:
        """按 data→metadata→原 arguments 优先级读取路径。"""
        data = result.get("data")
        metadata = result.get("metadata")
        path = None
        if isinstance(data, dict):
            path = data.get("path")
        if path is None and isinstance(metadata, dict):
            path = metadata.get("path")
        if isinstance(path, str) and path:
            return path
        arg_path = arguments.get("path")
        if isinstance(arg_path, str) and arg_path:
            return arg_path
        return "-"

    def _line_count(self, result: dict[str, Any]) -> int | None:
        """优先使用结构化 line_count，否则从 content 可见行数回退。"""
        data = result.get("data")
        if isinstance(data, dict):
            line_count = data.get("line_count")
            if isinstance(line_count, int):
                return line_count
        content = result.get("content")
        if isinstance(content, str):
            return len(content.splitlines())
        return None

    def _byte_size(self, result: dict[str, Any]) -> int | None:
        """优先 bytes，其次 chars，最后以 content UTF-8 长度回退。"""
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            byte_count = metadata.get("bytes")
            if isinstance(byte_count, int):
                return byte_count
            char_count = metadata.get("chars")
            if isinstance(char_count, int):
                return char_count
        content = result.get("content")
        if isinstance(content, str):
            return len(content.encode("utf-8"))
        return None


def format_bytes(size: int) -> str:
    """按十进制 B/KB/MB/GB 格式化非负字节数。"""
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("byte units must not be empty")


def preview_tool_output(value: str, max_chars: int) -> str:
    """短输出完整保留，长输出等分保留首尾并插入省略行。"""
    compact = value.strip()
    if len(compact) <= max_chars:
        return compact
    half = max_chars // 2
    return f"{compact[:half]}\n...\n{compact[-half:]}"
