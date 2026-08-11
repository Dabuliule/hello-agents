from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """一段保留原文件行号、语法类型和可选所属符号的索引文本。"""

    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    content: str


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    """从语法树抽取的可独立按名称检索的代码符号。"""

    name: str
    kind: str
    line: int
    signature: str


@dataclass(frozen=True, slots=True)
class ChunkedFile:
    """单个文件的语言识别、文本块与符号抽取结果。"""

    language: str
    chunks: tuple[CodeChunk, ...]
    symbols: tuple[CodeSymbol, ...]


_EXTENSIONS = {
    ".go": "go",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
}

_SYMBOL_NODES = {
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "javascript": {
        "class_declaration",
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
    },
    "python": {"class_definition", "function_definition"},
    "typescript": {
        "class_declaration",
        "function_declaration",
        "generator_function_declaration",
        "interface_declaration",
        "method_definition",
        "type_alias_declaration",
    },
    "tsx": {
        "class_declaration",
        "function_declaration",
        "generator_function_declaration",
        "interface_declaration",
        "method_definition",
        "type_alias_declaration",
    },
}


class TreeSitterChunker:
    """优先沿顶层符号边界、必要时按重叠行窗口切分源文件。"""

    def __init__(self, *, max_lines: int = 120, overlap_lines: int = 12) -> None:
        """配置每块最大行数及相邻长块的重叠行数。

        Raises:
            ValueError: 最大行数非正，或重叠不在 ``[0, max_lines)``。

        Example:
            ``TreeSitterChunker(max_lines=80, overlap_lines=8)`` 会让超长符号
            每 72 行开始一个新块，从而保留 8 行邻接上下文。
        """
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        if overlap_lines < 0 or overlap_lines >= max_lines:
            raise ValueError("overlap_lines must be between 0 and max_lines")
        self.max_lines = max_lines
        self.overlap_lines = overlap_lines
        self._languages = _load_languages()

    def chunk(self, path: Path, content: str) -> ChunkedFile:
        """解析文件并返回可检索块与所有嵌套符号。

        Args:
            path: 仅用后缀选择语言的源文件路径。
            content: UTF-8 解码后的完整文本。

        Returns:
            支持语言按最外层符号分块；未知语言退化为普通行窗口的结果。

        Example:
            ``chunk(Path("service.py"), "def run():\\n    pass")`` 会产生
            kind 为 ``function_definition``、symbol 为 ``run`` 的块。
        """
        language_name = _EXTENSIONS.get(path.suffix.casefold(), "text")
        language = self._languages.get(language_name)
        if language is None:
            return ChunkedFile(
                language=language_name,
                chunks=tuple(self._line_chunks(content, kind="text")),
                symbols=(),
            )

        source = content.encode("utf-8")
        tree = Parser(language).parse(source)
        symbol_nodes = list(_walk_symbol_nodes(tree.root_node, language_name))
        outer_nodes = _outermost_nodes(symbol_nodes)
        lines = content.splitlines()
        chunks: list[CodeChunk] = []
        cursor = 0
        for node in outer_nodes:
            start = node.start_point.row
            end = min(len(lines), node.end_point.row + 1)
            if start > cursor:
                chunks.extend(self._range_chunks(lines, cursor, start, kind="module"))
            name = _node_name(node, source)
            chunks.extend(
                self._range_chunks(
                    lines,
                    start,
                    end,
                    kind=node.type,
                    symbol=name,
                )
            )
            cursor = max(cursor, end)
        if cursor < len(lines):
            chunks.extend(self._range_chunks(lines, cursor, len(lines), kind="module"))
        if not chunks:
            chunks.extend(self._line_chunks(content, kind="module"))

        symbols = tuple(
            CodeSymbol(
                name=name,
                kind=node.type,
                line=node.start_point.row + 1,
                signature=_signature(lines, node.start_point.row),
            )
            for node in symbol_nodes
            if (name := _node_name(node, source))
        )
        return ChunkedFile(
            language=language_name,
            chunks=tuple(chunks),
            symbols=symbols,
        )

    def _line_chunks(self, content: str, *, kind: str) -> list[CodeChunk]:
        """把完整文本委托给通用行区间切分器。"""
        lines = content.splitlines()
        return self._range_chunks(lines, 0, len(lines), kind=kind)

    def _range_chunks(
        self,
        lines: list[str],
        start: int,
        end: int,
        *,
        kind: str,
        symbol: str | None = None,
    ) -> list[CodeChunk]:
        """将半开行区间切成带重叠、非空且使用 1-based 行号的块。"""
        chunks: list[CodeChunk] = []
        step = self.max_lines - self.overlap_lines
        position = start
        while position < end:
            chunk_end = min(position + self.max_lines, end)
            content = "\n".join(lines[position:chunk_end]).strip()
            if content:
                chunks.append(
                    CodeChunk(
                        start_line=position + 1,
                        end_line=chunk_end,
                        kind=kind,
                        symbol=symbol,
                        content=content,
                    )
                )
            if chunk_end >= end:
                break
            position += step
        return chunks


def _load_languages() -> dict[str, Language]:
    """延迟导入并构造项目支持的 Tree-sitter Language 对象。"""
    import tree_sitter_go
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript

    return {
        "go": Language(tree_sitter_go.language()),
        "javascript": Language(tree_sitter_javascript.language()),
        "python": Language(tree_sitter_python.language()),
        "typescript": Language(tree_sitter_typescript.language_typescript()),
        "tsx": Language(tree_sitter_typescript.language_tsx()),
    }


def _walk_symbol_nodes(node: Node, language: str) -> Any:
    """深度优先产生该语言定义为符号的全部语法节点。"""
    if node.type in _SYMBOL_NODES[language]:
        yield node
    for child in node.named_children:
        yield from _walk_symbol_nodes(child, language)


def _outermost_nodes(nodes: list[Node]) -> list[Node]:
    """过滤嵌套符号，仅保留用于划分互不重叠区间的最外层节点。"""
    ordered = sorted(nodes, key=lambda node: (node.start_byte, -node.end_byte))
    selected: list[Node] = []
    for node in ordered:
        if any(
            parent.start_byte <= node.start_byte and parent.end_byte >= node.end_byte
            for parent in selected
        ):
            continue
        selected.append(node)
    return selected


def _node_name(node: Node, source: bytes) -> str | None:
    """从通用 name 字段或 Go type_spec 中读取节点名称。"""
    name = node.child_by_field_name("name")
    if name is None and node.type == "type_declaration":
        name = next(
            (child for child in node.named_children if child.type == "type_spec"),
            None,
        )
        if name is not None:
            name = name.child_by_field_name("name")
    if name is None:
        return None
    return source[name.start_byte : name.end_byte].decode("utf-8", errors="replace")


def _signature(lines: list[str], row: int) -> str:
    """返回符号起始行最多 240 字符的单行签名。"""
    if row >= len(lines):
        return ""
    return lines[row].strip()[:240]
