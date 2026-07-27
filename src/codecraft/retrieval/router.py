from __future__ import annotations

import re
from dataclasses import dataclass

from codecraft.retrieval.models import RetrievalRequest

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")
_FILE_SUFFIXES = frozenset(
    {
        ".go",
        ".java",
        ".js",
        ".json",
        ".md",
        ".py",
        ".rs",
        ".toml",
        ".ts",
        ".tsx",
        ".yaml",
        ".yml",
    }
)
_QUESTION_WORDS = frozenset(
    {"find", "how", "locate", "what", "when", "where", "which", "who", "why"}
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    retrievers: tuple[str, ...]
    reason: str


class QueryRouter:
    """Build a deterministic, sequential retrieval plan from query shape."""

    def route(self, request: RetrievalRequest) -> RetrievalPlan:
        query = request.query.strip()
        retrievers: tuple[str, ...]
        if request.mode == "path":
            return RetrievalPlan(("lexical", "scan"), "path_mode")
        if _looks_like_path(query):
            return RetrievalPlan(("scan", "lexical"), "path_hint")
        if _IDENTIFIER.fullmatch(query):
            retrievers = (
                ("symbol", "scan")
                if request.case_sensitive
                else ("symbol", "lexical", "scan")
            )
            return RetrievalPlan(retrievers, "identifier")

        terms = query.split()
        first = terms[0].casefold() if terms else ""
        if _CJK.search(query) or len(terms) >= 4 or first in _QUESTION_WORDS:
            retrievers = ("scan",) if request.case_sensitive else ("lexical", "scan")
            return RetrievalPlan(retrievers, "natural_language")
        retrievers = ("scan",) if request.case_sensitive else ("scan", "lexical")
        return RetrievalPlan(retrievers, "exact_phrase")


def _looks_like_path(query: str) -> bool:
    if "/" in query or "\\" in query:
        return True
    folded = query.casefold()
    return any(folded.endswith(suffix) for suffix in _FILE_SUFFIXES)
