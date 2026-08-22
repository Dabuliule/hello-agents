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
    """按尝试顺序排列的 Retriever 名称及可解释路由原因。"""

    retrievers: tuple[str, ...]
    reason: str


class QueryRouter:
    """只按查询形态生成确定性顺序计划，不调用模型或执行检索。

    路由是低成本 heuristic，不做语义分类：标识符优先 symbol，自然语言优先 lexical，
    路径和精确短语优先 scan；大小写敏感内容绕开不支持该语义的 FTS5。真正的索引
    可用性和空结果处理留给 ContextEngine，因此缺少索引不会让 Auto 查询失败。
    """

    def route(self, request: RetrievalRequest) -> RetrievalPlan:
        """根据 mode、路径特征、标识符和自然语言形态生成确定性计划。

        Args:
            request: 尚未执行的标准检索请求。

        Returns:
            例如大小写不敏感标识符优先 ``symbol → lexical → scan``；
            路径提示优先 ``scan → lexical``。

        Example:
            >>> from pathlib import Path
            >>> request = RetrievalRequest(
            ...     query="PaymentGateway", root=Path("."), workspace_root=Path(".")
            ... )
            >>> QueryRouter().route(request).reason
            'identifier'
        """
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
    """识别含目录分隔符或常见源码/配置后缀的路径查询。"""
    if "/" in query or "\\" in query:
        return True
    folded = query.casefold()
    return any(folded.endswith(suffix) for suffix in _FILE_SUFFIXES)
