from __future__ import annotations

import asyncio
from pathlib import Path
import threading

import pytest

import codecraft.retrieval.retrievers.scan as scan_module
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


def test_scan_retriever_runs_blocking_walk_in_worker_thread(tmp_path):
    worker_threads: list[int] = []

    class ThreadRecordingRetriever(ScanRetriever):
        def _retrieve_sync(self, request):
            worker_threads.append(threading.get_ident())
            return super()._retrieve_sync(request)

    request = RetrievalRequest(
        query="missing",
        root=tmp_path,
        workspace_root=tmp_path,
    )
    main_thread = threading.get_ident()

    asyncio.run(ThreadRecordingRetriever().retrieve(request))

    assert worker_threads
    assert worker_threads[0] != main_thread


def test_scan_retriever_sorts_files_before_bounded_traversal(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "z-last.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "a-first.txt").write_text("needle\n", encoding="utf-8")
    real_scandir = scan_module.os.scandir
    with real_scandir(tmp_path) as scanner:
        reversed_entries = sorted(scanner, key=lambda item: item.name, reverse=True)

    class ReversedScandir:
        def __enter__(self):
            return iter(reversed_entries)

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(scan_module.os, "scandir", lambda path: ReversedScandir())
    request = RetrievalRequest(
        query="needle",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
    )

    response = asyncio.run(ScanRetriever().retrieve(request))

    assert [match.path for match in response.matches] == [
        "a-first.txt",
        "z-last.txt",
    ]


def test_scan_retriever_reports_only_actual_result_truncation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    request = RetrievalRequest(
        query="needle",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
        max_results=1,
    )

    exact = asyncio.run(ScanRetriever().retrieve(request))
    source.write_text("needle\nneedle again\n", encoding="utf-8")
    overflowing = asyncio.run(
        ScanRetriever(max_results=1).retrieve(
            RetrievalRequest(
                query="needle",
                root=tmp_path,
                workspace_root=tmp_path,
                mode="content",
                max_results=2,
            )
        )
    )

    assert len(exact.matches) == 1
    assert exact.truncated is False
    assert len(overflowing.matches) == 1
    assert overflowing.truncated is True
    assert overflowing.stats.skipped["result_limit"] == 1


def test_scan_retriever_enforces_file_and_total_byte_limits(tmp_path):
    for index in range(3):
        (tmp_path / f"source-{index}.txt").write_text("abcdef", encoding="utf-8")
    request = RetrievalRequest(
        query="missing",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
    )

    file_limited = asyncio.run(ScanRetriever(max_files=2).retrieve(request))
    byte_limited = asyncio.run(ScanRetriever(max_scanned_bytes=6).retrieve(request))

    assert file_limited.stats.candidate_file_count == 2
    assert file_limited.stats.skipped["file_limit"] == 1
    assert file_limited.truncated is True
    assert byte_limited.stats.candidate_file_count == 2
    assert byte_limited.stats.read_file_count == 1
    assert byte_limited.stats.scanned_bytes == 6
    assert byte_limited.stats.skipped["byte_limit"] == 1
    assert byte_limited.truncated is True


def test_scan_retriever_skips_file_removed_before_stat(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    original_stat = Path.stat

    def missing_stat(path, *args, **kwargs):
        if path == source:
            raise FileNotFoundError(source)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", missing_stat)
    request = RetrievalRequest(
        query="needle",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
    )

    response = asyncio.run(ScanRetriever().retrieve(request))

    assert response.matches == ()
    assert response.stats.skipped["unreadable"] == 1


def test_scan_retriever_skips_file_removed_before_read(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("needle\n", encoding="utf-8")
    original_open = Path.open

    def missing_open(path, *args, **kwargs):
        if path == source and args and args[0] == "rb":
            raise FileNotFoundError(source)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", missing_open)
    request = RetrievalRequest(
        query="needle",
        root=tmp_path,
        workspace_root=tmp_path,
        mode="content",
    )

    response = asyncio.run(ScanRetriever().retrieve(request))

    assert response.matches == ()
    assert response.stats.skipped["unreadable"] == 1


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
