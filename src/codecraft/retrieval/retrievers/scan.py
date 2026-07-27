from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codecraft.retrieval.files import (
    display_path,
    is_inside_workspace,
    iter_workspace_files,
    looks_binary,
)
from codecraft.retrieval.models import (
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStats,
)
from codecraft.retrieval.retrievers.base import Retriever


@dataclass(slots=True)
class _ScanState:
    matches: list[RetrievalMatch]
    skipped: dict[str, int]
    scanned_file_count: int = 0
    read_file_count: int = 0
    scanned_bytes: int = 0

    def is_full(self, max_results: int) -> bool:
        return len(self.matches) >= max_results


class ScanRetriever(Retriever):
    """Deterministic path and substring retrieval without a persistent index."""

    name = "scan"

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        files = (
            [request.root]
            if request.root.is_file()
            else iter_workspace_files(request.root)
        )
        query = request.query if request.case_sensitive else request.query.casefold()
        state = _ScanState(
            matches=[],
            skipped={"binary": 0, "large": 0, "escaped": 0},
        )

        for file_path in files:
            if state.is_full(request.max_results):
                break
            self._scan_file(file_path, request, query, state)

        return RetrievalResponse(
            matches=tuple(state.matches),
            stats=RetrievalStats(
                candidate_file_count=len(files),
                scanned_file_count=state.scanned_file_count,
                read_file_count=state.read_file_count,
                scanned_bytes=state.scanned_bytes,
                skipped=state.skipped,
            ),
            truncated=state.is_full(request.max_results),
        )

    def _scan_file(
        self,
        file_path: Path,
        request: RetrievalRequest,
        query: str,
        state: _ScanState,
    ) -> None:
        if not is_inside_workspace(file_path, request.workspace_roots):
            state.skipped["escaped"] += 1
            return

        state.scanned_file_count += 1
        visible_path = display_path(file_path, request.workspace_roots)
        self._append_path_match(visible_path, request, query, state)
        if state.is_full(request.max_results):
            return
        if request.mode not in {"both", "content"}:
            return

        stat = file_path.stat()
        if stat.st_size > request.max_file_bytes:
            state.skipped["large"] += 1
            return
        try:
            raw = file_path.read_bytes()
        except OSError:
            return

        state.read_file_count += 1
        state.scanned_bytes += len(raw)
        if looks_binary(raw):
            state.skipped["binary"] += 1
            return

        text = raw.decode("utf-8", errors="replace")
        self._append_content_matches(text, visible_path, request, query, state)

    @staticmethod
    def _append_path_match(
        visible_path: str,
        request: RetrievalRequest,
        query: str,
        state: _ScanState,
    ) -> None:
        if request.mode not in {"both", "path"}:
            return
        candidate_path = (
            visible_path if request.case_sensitive else visible_path.casefold()
        )
        if query in candidate_path:
            state.matches.append(RetrievalMatch(type="path", path=visible_path))

    def _append_content_matches(
        self,
        text: str,
        visible_path: str,
        request: RetrievalRequest,
        query: str,
        state: _ScanState,
    ) -> None:
        for line_number, line in enumerate(text.splitlines(), start=1):
            candidate_line = line if request.case_sensitive else line.casefold()
            if query not in candidate_line:
                continue
            state.matches.append(
                RetrievalMatch(
                    type="content",
                    path=visible_path,
                    line=line_number,
                    snippet=self._trim_line(line),
                )
            )
            if state.is_full(request.max_results):
                return

    @staticmethod
    def _trim_line(line: str, max_chars: int = 240) -> str:
        normalized = line.strip()
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[: max_chars - 1]}..."
