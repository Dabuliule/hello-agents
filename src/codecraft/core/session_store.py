from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
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
    SessionSource,
    SessionSummary,
)


@dataclass(frozen=True, slots=True)
class _SessionReadResult:
    """一次单遍日志扫描可选择收集的恢复配置、摘要字段和完整事件。"""

    config: SessionConfig | None
    cwd: Path | None
    source: SessionSource | None
    session_id: str | None
    created_at: datetime | None
    last_event_at: datetime | None
    event_count: int
    events: list[RuntimeEvent] | None


_SessionCandidate = tuple[Path, _SessionReadResult | None, SessionRestoreError | None]


class SessionStore:
    """基于 JSONL 文件的 session event 存储。

    每个 session 对应一个按日期分目录的 `.jsonl` 文件。恢复会话时不保存额外
    快照，而是重新读取事件日志并校验 seq 连续性。
    """

    SESSION_SCAN_CONCURRENCY = 8

    def __init__(self, codecraft_home: Path) -> None:
        """设置日期分片 sessions 根、路径缓存和按文件 append 锁。"""
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
        result = self._read_session_file(
            path,
            session_id=session_id,
            collect_events=True,
            require_restorable=False,
        )
        assert result.events is not None
        return result.events

    def _read_session_file(
        self,
        path: Path,
        *,
        session_id: str,
        collect_events: bool,
        require_restorable: bool,
    ) -> _SessionReadResult:
        """单遍读取 JSONL，逐行校验 schema、领域模型、session ID 和连续 seq。

        Args:
            path: 已定位的 session JSONL 文件。
            session_id: 文件名期望表达的 Session ID。
            collect_events: 是否保留完整 RuntimeEvent，列表扫描时可关闭节省内存。
            require_restorable: 是否要求首条为含有效 config 的 SESSION_STARTED。

        Returns:
            同时可服务 resume 和 list summary 的读取结果。

        Raises:
            SessionRestoreError: 编码/I/O、JSON、schema、config、ID 或 seq 任一
            信任检查失败。
        """
        events: list[RuntimeEvent] | None = [] if collect_events else None
        config: SessionConfig | None = None
        session_cwd: Path | None = None
        session_source: SessionSource | None = None
        started_session_id: str | None = None
        created_at: datetime | None = None
        last_event_at: datetime | None = None
        event_count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    data = self._decode_event_data(
                        stripped,
                        session_id=session_id,
                        path=path,
                        line_number=line_number,
                    )
                    self._validate_started_config_version(data, session_id)
                    if event_count == 0 and require_restorable:
                        validated_config = self._session_config_from_start(
                            data, session_id
                        )
                        session_cwd = validated_config.cwd
                        session_source = validated_config.source
                        if collect_events:
                            config = validated_config
                    event = self._event_from_data(
                        data,
                        session_id=session_id,
                        path=path,
                        line_number=line_number,
                    )
                    self._validate_event_position(
                        event,
                        session_id=session_id,
                        expected_seq=event_count + 1,
                    )
                    if started_session_id is None:
                        started_session_id = event.session_id
                        created_at = event.timestamp
                    last_event_at = event.timestamp
                    event_count += 1
                    if events is not None:
                        events.append(event)
        except (OSError, UnicodeError) as exc:
            raise SessionRestoreError(
                "failed to load session events",
                code="session_events_load_failed",
                metadata={"session_id": session_id, "path": str(path)},
            ) from exc

        if require_restorable and started_session_id is None:
            raise SessionRestoreError(
                "session contains no events",
                code="session_empty",
                metadata={"session_id": session_id},
            )

        return _SessionReadResult(
            config=config,
            cwd=session_cwd,
            source=session_source,
            session_id=started_session_id,
            created_at=created_at,
            last_event_at=last_event_at,
            event_count=event_count,
            events=events,
        )

    @staticmethod
    def _decode_event_data(
        line: str,
        *,
        session_id: str,
        path: Path,
        line_number: int,
    ) -> dict[str, Any]:
        """解析一行 JSON object 并在 Pydantic 前检查 RuntimeEvent schema version。"""
        metadata = {
            "session_id": session_id,
            "path": str(path),
            "line": line_number,
        }
        try:
            data = json.loads(line)
        except Exception as exc:
            raise SessionRestoreError(
                "failed to parse session event",
                code="session_event_parse_failed",
                metadata=metadata,
            ) from exc
        if not isinstance(data, dict):
            raise SessionRestoreError(
                "session event must be a JSON object",
                code="session_event_shape_invalid",
                metadata=metadata,
            )
        version = data.get("schema_version")
        if version != RUNTIME_EVENT_SCHEMA_VERSION:
            raise SessionRestoreError(
                "session event schema version is not supported",
                code="session_event_schema_unsupported",
                metadata={**metadata, "version": version},
            )
        return data

    @staticmethod
    def _event_from_data(
        data: dict[str, Any],
        *,
        session_id: str,
        path: Path,
        line_number: int,
    ) -> RuntimeEvent:
        """把原始 dict 严格验证为 RuntimeEvent，并附文件/行诊断。"""
        try:
            return RuntimeEvent.model_validate(data)
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

    async def load_raw_lines(self, session_id: str) -> list[str]:
        """在线程池读取原始 JSONL 行，供 trace/export 保留原始边界。"""
        return await asyncio.to_thread(self._load_raw_lines, session_id)

    def _load_raw_lines(self, session_id: str) -> list[str]:
        """同步读取原始行并归一化 I/O/编码失败。"""
        path = self._path_for_session(session_id)
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
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
        candidates = await self._scan_session_candidates(paths)
        for path, result, error in candidates:
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
            assert result is not None
            assert result.cwd is not None
            assert result.source is not None
            assert result.session_id is not None
            assert result.created_at is not None
            assert result.last_event_at is not None
            session_cwd = result.cwd

            if cwd_resolved and session_cwd != cwd_resolved:
                continue

            summaries.append(
                SessionSummary(
                    session_id=result.session_id,
                    path=path,
                    cwd=session_cwd,
                    source=result.source,
                    created_at=result.created_at,
                    last_event_at=result.last_event_at,
                    event_count=result.event_count,
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

    async def _scan_session_candidates(
        self,
        paths: list[Path],
    ) -> list[_SessionCandidate]:
        """最多八个 worker 并发扫描日志，同时保持输入 paths 的结果顺序。

        Worker 共享一个同步 iterator 领取索引；结果写入预分配槽位，因此文件
        完成先后不会改变 list_sessions 的后续排序/诊断行为。
        """
        for path in paths:
            self._paths.setdefault(path.stem, path)
        candidates: list[_SessionCandidate | None] = [None] * len(paths)
        pending = iter(enumerate(paths))

        async def load() -> None:
            """领取候选文件，在工作线程读取并把成功/恢复错误写回固定槽。"""
            for index, path in pending:
                try:
                    result = await asyncio.to_thread(
                        self._read_session_file,
                        path,
                        session_id=path.stem,
                        collect_events=False,
                        require_restorable=True,
                    )
                    candidates[index] = path, result, None
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
        """按最近事件时间尝试有效 Session，跳过并记录竞争期内损坏项。"""
        summaries = await self.list_sessions(cwd=cwd)
        attempts: list[dict[str, str]] = []
        for summary in summaries:
            try:
                return await self.resume(summary.session_id)
            except SessionRestoreError as exc:
                attempts.append({"session_id": summary.session_id, "code": exc.code})
        raise SessionRestoreError(
            "no session found to resume",
            code="session_not_found",
            metadata={"attempts": attempts},
        )

    async def resume(self, session_id: str) -> SessionSnapshot:
        """从事件日志恢复 session 配置和历史事件。"""
        path = self._path_for_session(session_id)
        result = await asyncio.to_thread(
            self._read_session_file,
            path,
            session_id=session_id,
            collect_events=True,
            require_restorable=True,
        )
        assert result.config is not None
        assert result.events is not None
        return SessionSnapshot(config=result.config, events=result.events)

    @staticmethod
    def _validate_config_version(config_data: dict[str, Any], session_id: str) -> None:
        """在构造 SessionConfig 前拒绝不受支持的持久化配置 schema。"""
        version = config_data.get("schema_version")
        if version != SESSION_CONFIG_SCHEMA_VERSION:
            raise SessionRestoreError(
                "session config schema version is not supported",
                code="session_config_schema_unsupported",
                metadata={"session_id": session_id, "version": version},
            )

    def _validate_started_config_version(
        self,
        event_data: dict[str, Any],
        session_id: str,
    ) -> None:
        """每次遇到 SESSION_STARTED 都预检其中 config version。"""
        if event_data.get("type") != RuntimeEventType.SESSION_STARTED.value:
            return
        payload = event_data.get("payload")
        config_data = payload.get("config") if isinstance(payload, dict) else None
        if isinstance(config_data, dict):
            self._validate_config_version(config_data, session_id)

    def _session_config_from_start(
        self,
        event_data: dict[str, Any],
        session_id: str,
    ) -> SessionConfig:
        """从首事件严格恢复配置，并验证 payload、schema 与 session_id 一致。"""
        if event_data.get("type") != RuntimeEventType.SESSION_STARTED.value:
            raise SessionRestoreError(
                "session log must start with session_started",
                code="session_start_event_missing",
                metadata={"session_id": session_id},
            )
        payload = event_data.get("payload")
        config_data = payload.get("config") if isinstance(payload, dict) else None
        if not isinstance(config_data, dict):
            raise SessionRestoreError(
                "session_started event is missing config payload",
                code="session_config_missing",
                metadata={"session_id": session_id},
            )
        self._validate_config_version(config_data, session_id)
        try:
            config = SessionConfig.model_validate(config_data)
        except (TypeError, ValueError) as exc:
            raise SessionRestoreError(
                "session config is invalid",
                code="session_config_validation_failed",
                metadata={"session_id": session_id},
            ) from exc
        if config.session_id != session_id:
            raise SessionRestoreError(
                "session config has mismatched session_id",
                code="session_config_id_mismatch",
                metadata={
                    "expected": session_id,
                    "actual": config.session_id,
                },
            )
        return config

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
        """确定性列出日期目录下全部 JSONL；目录不存在时返回空。"""
        if not self.sessions_dir.exists():
            return []
        return sorted(self.sessions_dir.glob("**/*.jsonl"))

    @staticmethod
    def _create_session_file(path: Path) -> None:
        """创建父目录和不可覆盖的空日志文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        """以二进制 append 写入一条 UTF-8 JSONL 并 flush 用户态缓冲。"""
        with path.open("ab") as handle:
            handle.write(f"{line}\n".encode())
            handle.flush()

    async def _append_line_serialized(
        self,
        path: Path,
        line: str,
        lock: asyncio.Lock,
    ) -> None:
        """用每文件 asyncio.Lock 串行化线程池 append，维持事件顺序。"""
        async with lock:
            await asyncio.to_thread(self._append_line, path, line)

    @classmethod
    def _invalid_file_metadata(cls, path: Path) -> tuple[int, datetime | None]:
        """为不可恢复日志提供尽力而为的非空行数和 mtime。"""
        return cls._count_raw_lines(path), cls._mtime(path)

    @staticmethod
    def _count_raw_lines(path: Path) -> int:
        """统计非空物理行；I/O 或编码失败返回 0。"""
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except (OSError, UnicodeError):
            return 0

    @staticmethod
    def _mtime(path: Path) -> datetime | None:
        """读取文件 mtime 并转换为 UTC；失败返回 None。"""
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            return None

    @staticmethod
    def _validate_event_position(
        event: RuntimeEvent,
        *,
        session_id: str,
        expected_seq: int,
    ) -> None:
        """验证每条事件属于目标 Session 且 seq 从 1 严格连续。"""
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
        if event.seq != expected_seq:
            raise SessionRestoreError(
                "session event sequence is not continuous",
                code="session_seq_not_continuous",
                metadata={
                    "session_id": session_id,
                    "expected": expected_seq,
                    "actual": event.seq,
                },
            )


class CodecraftFileNotFoundError(SessionRestoreError):
    """按 Session ID 无法定位持久化 JSONL 文件。"""
