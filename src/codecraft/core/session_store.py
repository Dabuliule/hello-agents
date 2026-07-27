from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codecraft.core.async_utils import finish_task_before_cancelling
from codecraft.core.errors import SessionError, SessionRestoreError
from codecraft.schema.event import (
    RUNTIME_EVENT_SCHEMA_VERSION,
    RuntimeEvent,
    RuntimeEventType,
)
from codecraft.schema.session import (
    SESSION_CONFIG_SCHEMA_VERSION,
    SessionConfig,
    SessionSnapshot,
    SessionSummary,
)


_SessionCandidate = tuple[
    Path,
    list[RuntimeEvent] | None,
    SessionRestoreError | None,
]


class SessionStore:
    """基于 JSONL 文件的 session event 存储。

    每个 session 对应一个按日期分目录的 `.jsonl` 文件。恢复会话时不保存额外
    快照，而是重新读取事件日志并校验 seq 连续性。
    """

    SESSION_SCAN_CONCURRENCY = 8

    def __init__(self, codecraft_home: Path) -> None:
        self.codecraft_home = codecraft_home.expanduser().resolve()
        self.sessions_dir = self.codecraft_home / "sessions"
        self._paths: dict[str, Path] = {}
        self._append_locks: dict[Path, asyncio.Lock] = {}

    async def create_session(self, config: SessionConfig) -> Path:
        """创建当前 session 的事件日志文件。"""
        created_at = config.created_at
        path = (
            self.sessions_dir
            / f"{created_at.year:04d}"
            / f"{created_at.month:02d}"
            / f"{created_at.day:02d}"
            / f"{config.session_id}.jsonl"
        )
        creation = asyncio.create_task(
            asyncio.to_thread(self._create_session_file, path)
        )
        try:
            await finish_task_before_cancelling(creation)
        except asyncio.CancelledError:
            self._paths[config.session_id] = path
            raise
        self._paths[config.session_id] = path
        return path

    async def append_event(self, event: RuntimeEvent) -> None:
        """追加单个事件到 session 日志，同时避免阻塞 runtime event loop。"""
        path = self._path_for_session(event.session_id)
        line = event.model_dump_json()
        append_lock = self._append_locks.setdefault(path, asyncio.Lock())
        append = asyncio.create_task(
            self._append_line_serialized(path, line, append_lock)
        )
        try:
            await finish_task_before_cancelling(append)
        except Exception as exc:
            raise SessionError(
                "failed to append session event",
                code="session_event_append_failed",
                metadata={
                    "session_id": event.session_id,
                    "path": str(path),
                    "event_type": event.type.value,
                    "seq": event.seq,
                    "cause": repr(exc),
                },
            ) from exc

    async def load_events(self, session_id: str) -> list[RuntimeEvent]:
        """读取并校验一个 session 的全部事件。"""
        return await asyncio.to_thread(self._load_events, session_id)

    def _load_events(self, session_id: str) -> list[RuntimeEvent]:
        """同步读取实现；只在线程池或同步诊断路径中调用。"""
        path = self._path_for_session(session_id)
        events: list[RuntimeEvent] = []

        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except Exception as exc:
                        raise SessionRestoreError(
                            "failed to parse session event",
                            code="session_event_parse_failed",
                            metadata={
                                "session_id": session_id,
                                "path": str(path),
                                "line": line_number,
                            },
                        ) from exc
                    if not isinstance(data, dict):
                        raise SessionRestoreError(
                            "session event must be a JSON object",
                            code="session_event_shape_invalid",
                            metadata={
                                "session_id": session_id,
                                "path": str(path),
                                "line": line_number,
                            },
                        )
                    version = data.get("schema_version")
                    if version != RUNTIME_EVENT_SCHEMA_VERSION:
                        raise SessionRestoreError(
                            "session event schema version is not supported",
                            code="session_event_schema_unsupported",
                            metadata={
                                "session_id": session_id,
                                "path": str(path),
                                "line": line_number,
                                "version": version,
                            },
                        )
                    try:
                        events.append(RuntimeEvent.model_validate(data))
                    except Exception as exc:
                        raise SessionRestoreError(
                            "failed to validate session event",
                            code="session_event_validation_failed",
                            metadata={
                                "session_id": session_id,
                                "path": str(path),
                                "line": line_number,
                            },
                        ) from exc
        except CodecraftFileNotFoundError as exc:
            raise exc
        except OSError as exc:
            raise SessionRestoreError(
                "failed to load session events",
                code="session_events_load_failed",
                metadata={"session_id": session_id, "path": str(path)},
            ) from exc

        self._validate_seq(session_id, events)
        return events

    async def load_raw_lines(self, session_id: str) -> list[str]:
        return await asyncio.to_thread(self._load_raw_lines, session_id)

    def _load_raw_lines(self, session_id: str) -> list[str]:
        path = self._path_for_session(session_id)
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SessionRestoreError(
                "failed to load raw session lines",
                code="session_raw_load_failed",
                metadata={"session_id": session_id, "path": str(path)},
            ) from exc

    async def list_sessions(
        self,
        cwd: Path | None = None,
        *,
        include_invalid: bool = False,
    ) -> list[SessionSummary]:
        """列出 session 摘要，可按 cwd 过滤。"""
        summaries: list[SessionSummary] = []
        cwd_resolved = cwd.expanduser().resolve() if cwd else None

        paths = await asyncio.to_thread(self._iter_session_files)
        candidates = await self._load_session_candidates(paths)
        for path, events, error in candidates:
            if error is not None:
                if include_invalid and cwd_resolved is None:
                    event_count, last_event_at = await asyncio.to_thread(
                        self._invalid_file_metadata,
                        path,
                    )
                    summaries.append(
                        SessionSummary(
                            session_id=path.stem,
                            path=path,
                            valid=False,
                            error_code=error.code,
                            error_message=error.message,
                            event_count=event_count,
                            last_event_at=last_event_at,
                        )
                    )
                continue
            assert events is not None
            if not events:
                summaries.append(
                    SessionSummary(
                        session_id=path.stem,
                        path=path,
                    )
                )
                continue

            first = events[0]
            last = events[-1]
            config = first.payload.get("config")
            if isinstance(config, dict):
                try:
                    self._validate_config_version(config, first.session_id)
                except SessionRestoreError as exc:
                    if include_invalid and cwd_resolved is None:
                        summaries.append(
                            SessionSummary(
                                session_id=path.stem,
                                path=path,
                                valid=False,
                                error_code=exc.code,
                                error_message=exc.message,
                                event_count=len(events),
                                last_event_at=last.timestamp,
                            )
                        )
                    continue
            else:
                config = {}
            session_cwd = self._optional_path(config.get("cwd"))

            if cwd_resolved and session_cwd != cwd_resolved:
                continue

            summaries.append(
                SessionSummary(
                    session_id=first.session_id,
                    path=path,
                    cwd=session_cwd,
                    source=config.get("source"),
                    created_at=first.timestamp,
                    last_event_at=last.timestamp,
                    event_count=len(events),
                )
            )

        return sorted(
            summaries,
            key=lambda summary: (
                summary.last_event_at
                or summary.created_at
                or datetime.min.replace(tzinfo=UTC)
            ),
            reverse=True,
        )

    async def _load_session_candidates(
        self,
        paths: list[Path],
    ) -> list[_SessionCandidate]:
        for path in paths:
            self._paths.setdefault(path.stem, path)
        candidates: list[_SessionCandidate | None] = [None] * len(paths)
        pending = iter(enumerate(paths))

        async def load() -> None:
            for index, path in pending:
                try:
                    candidates[index] = path, await self.load_events(path.stem), None
                except SessionRestoreError as exc:
                    candidates[index] = path, None, exc

        workers = [
            asyncio.create_task(load())
            for _ in range(min(self.SESSION_SCAN_CONCURRENCY, len(paths)))
        ]
        if workers:
            await asyncio.gather(*workers)
        return [candidate for candidate in candidates if candidate is not None]

    async def resume_last(self, cwd: Path | None = None) -> SessionSnapshot:
        summaries = await self.list_sessions(cwd=cwd)
        if not summaries:
            raise SessionRestoreError(
                "no session found to resume",
                code="session_not_found",
            )

        return await self.resume(summaries[0].session_id)

    async def resume(self, session_id: str) -> SessionSnapshot:
        """从事件日志恢复 session 配置和历史事件。"""
        events = await self.load_events(session_id)
        if not events:
            raise SessionRestoreError(
                "session contains no events",
                code="session_empty",
                metadata={"session_id": session_id},
            )

        started = events[0]
        if started.type != RuntimeEventType.SESSION_STARTED:
            raise SessionRestoreError(
                "session log must start with session_started",
                code="session_start_event_missing",
                metadata={"session_id": session_id},
            )

        config_data = started.payload.get("config")
        if not isinstance(config_data, dict):
            raise SessionRestoreError(
                "session_started event is missing config payload",
                code="session_config_missing",
                metadata={"session_id": session_id},
            )
        self._validate_config_version(config_data, session_id)

        return SessionSnapshot(
            config=SessionConfig.model_validate(config_data),
            events=events,
        )

    @staticmethod
    def _validate_config_version(config_data: dict[str, Any], session_id: str) -> None:
        version = config_data.get("schema_version")
        if version != SESSION_CONFIG_SCHEMA_VERSION:
            raise SessionRestoreError(
                "session config schema version is not supported",
                code="session_config_schema_unsupported",
                metadata={"session_id": session_id, "version": version},
            )

    def _path_for_session(self, session_id: str) -> Path:
        """定位 session 日志路径，并缓存 glob 的结果。"""
        if session_id in self._paths:
            return self._paths[session_id]

        matches = list(self.sessions_dir.glob(f"**/{session_id}.jsonl"))
        if not matches:
            raise CodecraftFileNotFoundError(
                "session file not found",
                code="session_file_not_found",
                metadata={"session_id": session_id},
            )

        path = matches[0]
        self._paths[session_id] = path
        return path

    def _iter_session_files(self) -> list[Path]:
        if not self.sessions_dir.exists():
            return []
        return sorted(self.sessions_dir.glob("**/*.jsonl"))

    @staticmethod
    def _create_session_file(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        with path.open("ab") as handle:
            handle.write(f"{line}\n".encode())
            handle.flush()

    async def _append_line_serialized(
        self,
        path: Path,
        line: str,
        lock: asyncio.Lock,
    ) -> None:
        async with lock:
            await asyncio.to_thread(self._append_line, path, line)

    @classmethod
    def _invalid_file_metadata(cls, path: Path) -> tuple[int, datetime | None]:
        return cls._count_raw_lines(path), cls._mtime(path)

    @staticmethod
    def _count_raw_lines(path: Path) -> int:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    @staticmethod
    def _mtime(path: Path) -> datetime | None:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            return None

    @staticmethod
    def _validate_seq(session_id: str, events: list[RuntimeEvent]) -> None:
        """确保事件属于同一个 session，且 seq 从 1 开始连续递增。"""
        for expected, event in enumerate(events, start=1):
            if event.session_id != session_id:
                raise SessionRestoreError(
                    "session event has mismatched session_id",
                    code="session_id_mismatch",
                    metadata={
                        "expected": session_id,
                        "actual": event.session_id,
                        "seq": event.seq,
                    },
                )

            if event.seq != expected:
                raise SessionRestoreError(
                    "session event sequence is not continuous",
                    code="session_seq_not_continuous",
                    metadata={
                        "session_id": session_id,
                        "expected": expected,
                        "actual": event.seq,
                    },
                )

    @staticmethod
    def _optional_path(value: object) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        return Path(value).expanduser().resolve()


class CodecraftFileNotFoundError(SessionRestoreError):
    pass
