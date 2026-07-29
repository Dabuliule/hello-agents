from __future__ import annotations

import asyncio

import pytest

from codecraft.retrieval import (
    ContextEngine,
    QueryRouter,
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStats,
    RetrievalUnavailableError,
    Retriever,
    ScanRetriever,
)


class StaticRetriever(Retriever):
    name = "static"

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return RetrievalResponse(
            matches=(
                RetrievalMatch(
                    type="content",
                    path="virtual/result.py",
                    line=7,
                    snippet=request.query,
                ),
            ),
            stats=RetrievalStats(candidate_file_count=1, scanned_file_count=1),
        )


def test_scan_retriever_preserves_path_content_and_costs(tmp_path):
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def build_agent():\n    return 'ready'\n", encoding="utf-8")
    ignored = tmp_path / "__pycache__" / "agent.py"
    ignored.parent.mkdir()
    ignored.write_text("def build_agent(): pass\n", encoding="utf-8")
    request = RetrievalRequest(
        query="agent",
        root=tmp_path,
        workspace_root=tmp_path,
    )

    response = asyncio.run(ScanRetriever().retrieve(request))

    assert [match.as_dict() for match in response.matches] == [
        {"type": "path", "path": "src/agent.py"},
        {
            "type": "content",
            "path": "src/agent.py",
            "line": 1,
            "snippet": "def build_agent():",
        },
    ]
    assert response.stats.candidate_file_count == 1
    assert response.stats.scanned_file_count == 1
    assert response.stats.read_file_count == 1
    assert response.stats.scanned_bytes == source.stat().st_size


def test_scan_retriever_uses_workspace_relative_paths_for_scoped_search(tmp_path):
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("class Agent: pass\n", encoding="utf-8")
    request = RetrievalRequest(
        query="Agent",
        root=source.parent,
        workspace_root=tmp_path,
        mode="content",
    )

    response = asyncio.run(ScanRetriever().retrieve(request))

    assert response.matches[0].path == "src/agent.py"


def test_scan_retriever_rejects_scope_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    request = RetrievalRequest(
        query="secret",
        root=outside,
        workspace_root=workspace,
    )

    with pytest.raises(RetrievalUnavailableError, match="outside workspace"):
        asyncio.run(ScanRetriever().retrieve(request))


def test_context_engine_selects_configured_retrievers(tmp_path):
    request = RetrievalRequest(
        query="needle",
        root=tmp_path,
        workspace_root=tmp_path,
    )
    engine = ContextEngine(
        [ScanRetriever(), StaticRetriever()],
        default_retriever="static",
    )

    default_response = asyncio.run(engine.retrieve(request))
    scan_response = asyncio.run(engine.retrieve(request, retriever_name="scan"))

    assert engine.retriever_names == ("scan", "static")
    assert default_response.matches[0].path == "virtual/result.py"
    assert scan_response.matches == ()
    with pytest.raises(ValueError, match="unknown retriever: missing"):
        asyncio.run(engine.retrieve(request, retriever_name="missing"))


def test_context_engine_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="unique names"):
        ContextEngine([ScanRetriever(), ScanRetriever()])
    with pytest.raises(ValueError, match="unknown default retriever"):
        ContextEngine([ScanRetriever()], default_retriever="missing")


def test_query_router_builds_deterministic_sequential_plans(tmp_path):
    router = QueryRouter()

    def route(query: str, *, mode="content", case_sensitive=False):
        return router.route(
            RetrievalRequest(
                query=query,
                root=tmp_path,
                workspace_root=tmp_path,
                mode=mode,
                case_sensitive=case_sensitive,
            )
        )

    assert route("PaymentGateway").retrievers == ("symbol", "lexical", "scan")
    assert route("PaymentGateway").reason == "identifier"
    assert route("where are permissions checked").retrievers == (
        "lexical",
        "scan",
    )
    assert route("权限在哪里检查").retrievers == ("lexical", "scan")
    assert route("权限在哪里检查").reason == "natural_language"
    assert route("retry budget exhausted").retrievers == ("scan", "lexical")
    assert route("src/auth/service.py").retrievers == ("scan", "lexical")
    assert route("service.py").reason == "path_hint"
    assert route("invoice", mode="path").retrievers == ("lexical", "scan")
    assert route("ExactName", case_sensitive=True).retrievers == ("symbol", "scan")


def test_context_engine_auto_route_skips_unconfigured_retrievers(tmp_path):
    source = tmp_path / "agent.py"
    source.write_text("class Agent: pass\n", encoding="utf-8")
    engine = ContextEngine([ScanRetriever()])
    request = RetrievalRequest(
        query="Agent",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
    )

    response = asyncio.run(engine.retrieve(request, retriever_name="auto"))

    assert response.retriever == "scan"
    assert response.route_reason == "identifier"
    assert response.attempted_retrievers == ("scan",)
    assert response.matches[0].path == "agent.py"
