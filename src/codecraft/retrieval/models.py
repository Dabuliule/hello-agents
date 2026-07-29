from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RetrievalMode = Literal["both", "content", "path"]
MatchType = Literal["content", "path"]


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    root: Path
    workspace_root: Path
    mode: RetrievalMode = "both"
    case_sensitive: bool = False
    max_results: int = 100
    max_file_bytes: int = 1_000_000


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    type: MatchType
    path: str
    line: int | None = None
    snippet: str | None = None

    def as_dict(self) -> dict[str, object]:
        match: dict[str, object] = {"type": self.type, "path": self.path}
        if self.line is not None:
            match["line"] = self.line
        if self.snippet is not None:
            match["snippet"] = self.snippet
        return match


@dataclass(frozen=True, slots=True)
class RetrievalStats:
    candidate_file_count: int = 0
    scanned_file_count: int = 0
    read_file_count: int = 0
    scanned_bytes: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    matches: tuple[RetrievalMatch, ...]
    stats: RetrievalStats
    truncated: bool = False
    retriever: str | None = None
    fallback_from: str | None = None
    route_reason: str | None = None
    attempted_retrievers: tuple[str, ...] = ()
