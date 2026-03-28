"""基于 psycopg v3 与连接池的 PostgreSQL 情景记忆存储实现。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .embedding_service import EmbeddingService
from .models import Action, Episode
from .qdrant_episode_vector_store import QdrantEpisodeVectorStore

logger = logging.getLogger(__name__)

# 向量数据不在本库持久化，统一交由外部向量系统（如 Qdrant）管理。
CREATE_EPISODES_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    result TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    reflection TEXT,
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0,
    user_id TEXT,
    tags TEXT[]
)
"""

CREATE_EPISODE_ACTIONS_SQL = """
CREATE TABLE IF NOT EXISTS episode_actions (
    episode_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_input JSONB NOT NULL,
    tool_output JSONB NOT NULL,
    PRIMARY KEY (episode_id, step),
    CONSTRAINT fk_episode_actions_episode
        FOREIGN KEY (episode_id)
        REFERENCES episodes (episode_id)
        ON DELETE CASCADE
)
"""

CREATE_INDEXES_SQL: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes (success)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_success_created ON episodes (success, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_last_accessed ON episodes (last_accessed_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_tags ON episodes USING GIN (tags)",
    "CREATE INDEX IF NOT EXISTS idx_episode_actions_episode_id ON episode_actions (episode_id)",
)


class EpisodeNotFoundError(Exception):
    """当记录不存在时抛出。"""


class PostgresEpisodeStore:
    def __init__(
        self,
        dsn: str,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[QdrantEpisodeVectorStore] = None,
    ):
        self._dsn = dsn
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self.pool = ConnectionPool(
            conninfo=self._dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self.pool.open()

    def close(self) -> None:
        self.pool.close()

    def create_tables(self) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(CREATE_EPISODES_SQL)
                    cur.execute(CREATE_EPISODE_ACTIONS_SQL)
                    for sql in CREATE_INDEXES_SQL:
                        cur.execute(sql)

    def insert_episode(self, episode: Episode) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._execute_upsert_episode(cur=cur, episode=episode)

    def insert_actions(self, episode_id: str, actions: List[Action]) -> None:
        if not actions:
            return

        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._execute_upsert_actions(cur=cur, episode_id=episode_id, actions=actions)

    def insert_full_episode(self, episode: Episode, actions: List[Action]) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    self._execute_upsert_episode(cur=cur, episode=episode)
                    if actions:
                        self._execute_upsert_actions(cur=cur, episode_id=episode.episode_id, actions=actions)
        self._try_upsert_vector(episode)

    def touch_episode(self, episode_id: str) -> None:
        sql = """
        UPDATE episodes
        SET
            access_count = LEAST(access_count + 1, 1000),
            last_accessed_at = NOW()
        WHERE episode_id = %s
        """
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(sql, (episode_id,))
                    if cur.rowcount == 0:
                        raise EpisodeNotFoundError(f"未找到 episode: {episode_id}")

    def query_episodes(
        self,
        session_id: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        match_all_tags: bool = False,
    ) -> List[Episode]:
        if limit <= 0:
            return []

        where_clauses: list[str] = []
        params: list[Any] = []

        if session_id is not None:
            where_clauses.append("session_id = %s")
            params.append(session_id)

        if success is not None:
            where_clauses.append("success = %s")
            params.append(success)

        if start_time is not None:
            where_clauses.append("created_at >= %s")
            params.append(start_time)

        if end_time is not None:
            where_clauses.append("created_at <= %s")
            params.append(end_time)

        if tags:
            if match_all_tags:
                where_clauses.append("tags @> %s")
            else:
                where_clauses.append("tags && %s")
            params.append(tags)

        query_sql = """
        SELECT
            episode_id,
            session_id,
            query,
            result,
            success,
            score,
            reflection,
            importance,
            created_at,
            last_accessed_at,
            access_count,
            user_id,
            tags
        FROM episodes
        """
        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)

        query_sql += " ORDER BY success DESC, importance DESC, created_at DESC LIMIT %s"
        params.append(limit)

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, tuple(params))
                rows = cur.fetchall()

        return [self._episode_from_row(row) for row in rows]

    def get_episode_with_actions(self, episode_id: str) -> Tuple[Episode, List[Action]]:
        episode_sql = """
        SELECT
            episode_id,
            session_id,
            query,
            result,
            success,
            score,
            reflection,
            importance,
            created_at,
            last_accessed_at,
            access_count,
            user_id,
            tags
        FROM episodes
        WHERE episode_id = %s
        """
        actions_sql = """
        SELECT
            step,
            tool_name,
            tool_input,
            tool_output
        FROM episode_actions
        WHERE episode_id = %s
        ORDER BY step ASC
        """

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(episode_sql, (episode_id,))
                row = cur.fetchone()
                if row is None:
                    raise EpisodeNotFoundError(f"未找到 episode: {episode_id}")
                episode = self._episode_from_row(row)

                cur.execute(actions_sql, (episode_id,))
                action_rows = cur.fetchall()

        actions = [self._action_from_row(row) for row in action_rows]
        return episode, actions

    def update_importance(self, episode_id: str) -> None:
        # 按 7 天半衰期计算时间衰减：exp(-ln(2) * 经过天数 / 7)
        sql = """
        UPDATE episodes
        SET importance = LEAST(
            1.0,
            GREATEST(
                0.0,
                (
                    0.45
                    + LEAST(access_count, 1000) * 0.0003
                    + CASE WHEN success THEN 0.12 ELSE -0.06 END
                    + EXP(-LN(2) * (GREATEST(0, EXTRACT(EPOCH FROM (NOW() - created_at))) / 86400.0) / 7.0) * 0.2
                )
            )
        )
        WHERE episode_id = %s
        """
        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(sql, (episode_id,))
                    if cur.rowcount == 0:
                        raise EpisodeNotFoundError(f"未找到 episode: {episode_id}")

    @staticmethod
    def _execute_upsert_episode(cur: Any, episode: Episode) -> None:
        sql = """
        INSERT INTO episodes (
            episode_id,
            session_id,
            query,
            result,
            success,
            score,
            reflection,
            importance,
            created_at,
            last_accessed_at,
            access_count,
            user_id,
            tags
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (episode_id) DO UPDATE SET
            session_id = EXCLUDED.session_id,
            query = EXCLUDED.query,
            result = EXCLUDED.result,
            success = EXCLUDED.success,
            score = EXCLUDED.score,
            reflection = EXCLUDED.reflection,
            importance = EXCLUDED.importance,
            created_at = EXCLUDED.created_at,
            last_accessed_at = EXCLUDED.last_accessed_at,
            access_count = EXCLUDED.access_count,
            user_id = EXCLUDED.user_id,
            tags = EXCLUDED.tags
        """
        params = (
            episode.episode_id,
            episode.session_id,
            episode.query,
            episode.result,
            episode.success,
            float(episode.score),
            episode.reflection,
            float(episode.importance),
            episode.created_at,
            episode.last_accessed_at,
            int(episode.access_count),
            episode.user_id,
            episode.tags or [],
        )
        cur.execute(sql, params)

    @staticmethod
    def _execute_upsert_actions(cur: Any, episode_id: str, actions: List[Action]) -> None:
        sql = """
        INSERT INTO episode_actions (
            episode_id,
            step,
            tool_name,
            tool_input,
            tool_output
        ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (episode_id, step) DO UPDATE SET
            tool_name = EXCLUDED.tool_name,
            tool_input = EXCLUDED.tool_input,
            tool_output = EXCLUDED.tool_output
        """
        values: list[tuple[Any, ...]] = [
            (
                episode_id,
                int(action.step),
                action.tool_name,
                Jsonb(action.tool_input),
                Jsonb(action.tool_output),
            )
            for action in actions
        ]
        cur.executemany(sql, values)

    @staticmethod
    def _episode_from_row(row: dict[str, Any]) -> Episode:
        created_at_value = row["created_at"]
        if created_at_value.tzinfo is None:
            created_at_value = created_at_value.replace(tzinfo=timezone.utc)

        last_accessed = row.get("last_accessed_at")
        if last_accessed is not None and last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        tags_value = row.get("tags") or []
        normalized_tags = [str(tag) for tag in tags_value]

        return Episode(
            episode_id=str(row["episode_id"]),
            session_id=str(row["session_id"]),
            query=str(row["query"]),
            result=str(row["result"]),
            success=bool(row["success"]),
            score=float(row["score"]),
            reflection=row.get("reflection"),
            importance=float(row["importance"]),
            created_at=created_at_value,
            last_accessed_at=last_accessed,
            access_count=int(row.get("access_count") or 0),
            user_id=row.get("user_id"),
            tags=normalized_tags,
        )

    @staticmethod
    def _action_from_row(row: dict[str, Any]) -> Action:
        return Action(
            step=int(row["step"]),
            tool_name=str(row["tool_name"]),
            tool_input=dict(row["tool_input"] or {}),
            tool_output=dict(row["tool_output"] or {}),
        )

    def _try_upsert_vector(self, episode: Episode) -> None:
        if self._embedding_service is None or self._vector_store is None:
            return
        try:
            text = self._build_embedding_text(episode)
            embedding = self._embedding_service.embed(text)
            self._vector_store.upsert(episode_id=episode.episode_id, embedding=embedding)
        except Exception as exc:
            # 向量写入失败不影响 PostgreSQL 主流程。
            logger.warning("Qdrant 向量写入失败，已忽略: episode_id=%s, error=%s", episode.episode_id, exc)

    @staticmethod
    def _build_embedding_text(episode: Episode) -> str:
        reflection = episode.reflection or ""
        return "\n".join([episode.query, episode.result, reflection]).strip()

