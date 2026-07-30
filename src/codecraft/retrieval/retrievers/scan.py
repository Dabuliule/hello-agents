from __future__ import annotations

import asyncio
from collections.abc import Generator
from dataclasses import dataclass
import os
from pathlib import Path

from codecraft.retrieval.errors import RetrievalUnavailableError
from codecraft.retrieval.files import (
    SKIPPED_NAMES,
    display_path,
    is_inside_workspace,
    looks_binary,
)
from codecraft.retrieval.models import (
    RetrievalMatch,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalStats,
)
from codecraft.retrieval.retrievers.base import Retriever

_DEFAULT_MAX_FILES = 10_000
_DEFAULT_MAX_SCANNED_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_RESULTS = 1_000


@dataclass(slots=True)
class _ScanState:
    matches: list[RetrievalMatch]
    skipped: dict[str, int]
    candidate_file_count: int = 0
    scanned_file_count: int = 0
    read_file_count: int = 0
    scanned_bytes: int = 0
    limit_reached: bool = False

    def has_extra_result(self, max_results: int) -> bool:
        return len(self.matches) > max_results


class ScanRetriever(Retriever):
    """Bounded path and substring retrieval without a persistent index."""

    name = "scan"

    def __init__(
        self,
        *,
        max_files: int = _DEFAULT_MAX_FILES,
        max_scanned_bytes: int = _DEFAULT_MAX_SCANNED_BYTES,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> None:
        if max_files < 1:
            raise ValueError("max_files must be at least 1")
        if max_scanned_bytes < 1:
            raise ValueError("max_scanned_bytes must be at least 1")
        if max_results < 1:
            raise ValueError("max_results must be at least 1")
        self.max_files = max_files
        self.max_scanned_bytes = max_scanned_bytes
        self.max_results = max_results

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        return await asyncio.to_thread(self._retrieve_sync, request)

    def _retrieve_sync(self, request: RetrievalRequest) -> RetrievalResponse:
        if not is_inside_workspace(request.root, request.workspace_root):
            raise RetrievalUnavailableError("request root is outside workspace")
        query = request.query if request.case_sensitive else request.query.casefold()
        result_limit = min(request.max_results, self.max_results)
        state = _ScanState(
            matches=[],
            skipped={
                "binary": 0,
                "large": 0,
                "escaped": 0,
                "unreadable": 0,
                "file_limit": 0,
                "byte_limit": 0,
                "result_limit": 0,
            },
        )

        files = self._iter_candidate_files(request.root, state)
        try:
            for file_path in files:
                if state.candidate_file_count >= self.max_files:
                    state.skipped["file_limit"] += 1
                    state.limit_reached = True
                    break
                state.candidate_file_count += 1
                if not self._scan_file(
                    file_path,
                    request,
                    query,
                    state,
                    result_limit=result_limit,
                ):
                    break
                if state.has_extra_result(result_limit):
                    break
        finally:
            files.close()

        result_truncated = state.has_extra_result(result_limit)
        if result_truncated:
            state.skipped["result_limit"] = 1

        return RetrievalResponse(
            matches=tuple(state.matches[:result_limit]),
            stats=RetrievalStats(
                candidate_file_count=state.candidate_file_count,
                scanned_file_count=state.scanned_file_count,
                read_file_count=state.read_file_count,
                scanned_bytes=state.scanned_bytes,
                skipped=state.skipped,
            ),
            truncated=result_truncated or state.limit_reached,
        )

    def _iter_candidate_files(
        self,
        root: Path,
        state: _ScanState,
    ) -> Generator[Path, None, None]:
        try:
            if root.is_file():
                yield root
                return
            if not root.is_dir():
                state.skipped["unreadable"] += 1
                return
        except OSError:
            state.skipped["unreadable"] += 1
            return
        yield from self._walk_directory(root, state)

    def _walk_directory(
        self,
        directory: Path,
        state: _ScanState,
    ) -> Generator[Path, None, None]:
        try:
            with os.scandir(directory) as scanner:
                entries = sorted(scanner, key=lambda item: item.name)
        except OSError:
            state.skipped["unreadable"] += 1
            return

        for item in entries:
            if item.name in SKIPPED_NAMES:
                continue
            path = Path(item.path)
            try:
                if item.is_file(follow_symlinks=False):
                    yield path
                elif item.is_dir(follow_symlinks=False):
                    yield from self._walk_directory(path, state)
                elif item.is_symlink() and path.is_file():
                    yield path
            except OSError:
                state.skipped["unreadable"] += 1

    def _scan_file(
        self,
        file_path: Path,
        request: RetrievalRequest,
        query: str,
        state: _ScanState,
        *,
        result_limit: int,
    ) -> bool:
        if not is_inside_workspace(file_path, request.workspace_root):
            state.skipped["escaped"] += 1
            return True

        state.scanned_file_count += 1
        visible_path = display_path(file_path, request.workspace_root)
        self._append_path_match(visible_path, request, query, state)
        if state.has_extra_result(result_limit):
            return True
        if request.mode not in {"both", "content"}:
            return True

        raw = self._read_bounded_file(file_path, request, state)
        if raw is None:
            return not state.limit_reached
        if looks_binary(raw):
            state.skipped["binary"] += 1
            return True

        text = raw.decode("utf-8", errors="replace")
        self._append_content_matches(
            text,
            visible_path,
            request,
            query,
            state,
            result_limit=result_limit,
        )
        return True

    def _read_bounded_file(
        self,
        file_path: Path,
        request: RetrievalRequest,
        state: _ScanState,
    ) -> bytes | None:
        try:
            stat = file_path.stat()
        except OSError:
            state.skipped["unreadable"] += 1
            return None
        if stat.st_size > request.max_file_bytes:
            state.skipped["large"] += 1
            return None

        remaining_bytes = self.max_scanned_bytes - state.scanned_bytes
        if remaining_bytes <= 0 or stat.st_size > remaining_bytes:
            state.skipped["byte_limit"] += 1
            state.limit_reached = True
            return None

        read_limit = min(request.max_file_bytes, remaining_bytes)
        try:
            with file_path.open("rb") as stream:
                raw = stream.read(read_limit + 1)
        except OSError:
            state.skipped["unreadable"] += 1
            return None

        state.read_file_count += 1
        observed = raw[:read_limit]
        state.scanned_bytes += len(observed)
        if len(raw) > remaining_bytes:
            state.skipped["byte_limit"] += 1
            state.limit_reached = True
            return None
        if len(raw) > request.max_file_bytes:
            state.skipped["large"] += 1
            return None
        return observed

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
        *,
        result_limit: int,
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
            if state.has_extra_result(result_limit):
                return

    @staticmethod
    def _trim_line(line: str, max_chars: int = 240) -> str:
        normalized = line.strip()
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[: max_chars - 1]}..."
