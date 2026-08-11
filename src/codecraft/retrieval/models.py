from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RetrievalMode = Literal["both", "content", "path"]
MatchType = Literal["content", "path"]


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """一次检索的查询、文件作用域、匹配模式和资源预算。"""

    query: str
    root: Path
    workspace_root: Path
    mode: RetrievalMode = "both"
    case_sensitive: bool = False
    max_results: int = 100
    max_file_bytes: int = 1_000_000


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """返回给工具层的路径命中或带行号的内容命中。"""

    type: MatchType
    path: str
    line: int | None = None
    snippet: str | None = None

    def as_dict(self) -> dict[str, object]:
        """序列化存在的字段，避免路径命中携带无意义的 null。

        Example:
            >>> RetrievalMatch(type="path", path="src/api.py").as_dict()
            {'type': 'path', 'path': 'src/api.py'}
        """
        match: dict[str, object] = {"type": self.type, "path": self.path}
        if self.line is not None:
            match["line"] = self.line
        if self.snippet is not None:
            match["snippet"] = self.snippet
        return match


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    """检索成本与按原因跳过文件的可观测统计。"""

    candidate_file_count: int = 0
    scanned_file_count: int = 0
    read_file_count: int = 0
    scanned_bytes: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """标准化命中、成本、截断和路由/降级事实。"""

    matches: tuple[RetrievalMatch, ...]
    stats: RetrievalStats
    truncated: bool = False
    retriever: str | None = None
    fallback_from: str | None = None
    route_reason: str | None = None
    attempted_retrievers: tuple[str, ...] = ()
