"""EpisodicMemory 实现。"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from memory.base import MemoryBase, MemoryRecord
from memory.storage import Action, Episode, EpisodeNotFoundError, PostgresEpisodeStore


@dataclass(frozen=True)
class EpisodicMemoryRecord(MemoryRecord):
    """情景记忆记录：使用结构化字段表达经验，不复用文本 content 字段。"""

    session_id: str = ""
    query: str = ""
    result: str = ""
    success: bool = False
    score: float = 0.0
    reflection: Optional[str] = None
    access_count: int = 0
    actions: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 不能为空")


class EpisodicMemory(MemoryBase):
    """情景记忆：基于 PostgreSQL 持久化，向量写入由 store 内部链路处理。"""

    def __init__(
            self,
            store: PostgresEpisodeStore,
            default_session_id: str = "default",
            retrieve_window: int = 200,
    ) -> None:
        if not default_session_id.strip():
            raise ValueError("default_session_id 不能为空")
        if retrieve_window < 1:
            raise ValueError("retrieve_window 必须大于 0")

        self._store = store
        self._default_session_id = default_session_id
        self._retrieve_window = retrieve_window

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        try:
            self._store.touch_episode(record_id)
            episode, actions = self._store.get_episode_with_actions(record_id)
        except EpisodeNotFoundError:
            return None
        return self._episode_to_record(episode=episode, actions=actions)

    def list(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        if limit is not None and limit <= 0:
            return []
        fetch_limit = limit if limit is not None else self._retrieve_window
        episodes = self._store.query_episodes(limit=fetch_limit)
        return [self._episode_to_record(episode=ep, actions=[]) for ep in episodes]

    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryRecord]:
        if limit <= 0:
            return []

        normalized_query = (query or "").strip()
        if not normalized_query:
            episodes = self._store.query_episodes(limit=limit)
            return self._build_records_with_actions(episodes)

        recall_limit = max(limit * 3, 20)
        recalled = self._store.search_similar_episodes(query=normalized_query, limit=recall_limit)
        scored = [(self._score_episode(item), item) for item in recalled]
        scored.sort(key=lambda x: x[0], reverse=True)

        records: List[MemoryRecord] = []
        for _, (episode, actions, _) in scored[:limit]:
            records.append(self._episode_to_record(episode=episode, actions=actions))
        return records

    def delete(self, record_id: str) -> bool:
        return self._store.delete_episode(record_id)

    def clear(self) -> None:
        self._store.clear_episodes()

    def add(self, episode: Episode, actions: Optional[List[Action]] = None) -> str:
        # 统一由内部生成主键，忽略外部传入的 episode_id。
        episode.episode_id = f"ep_{uuid.uuid4().hex}"
        action_list = actions or []
        self._store.insert_full_episode(episode=episode, actions=action_list)
        return episode.episode_id

    def _build_records_with_actions(self, episodes: List[Episode]) -> List[MemoryRecord]:
        records: List[MemoryRecord] = []
        for episode in episodes:
            try:
                full_episode, actions = self._store.get_episode_with_actions(episode.episode_id)
            except EpisodeNotFoundError:
                continue
            records.append(self._episode_to_record(episode=full_episode, actions=actions))
        return records

    @staticmethod
    def _score_episode(item: Tuple[Episode, List[Action], float]) -> float:
        try:
            episode, _, semantic_score = item
            safe_semantic = min(max(float(semantic_score), 0.0), 1.0)
            safe_episode_score = min(max(float(episode.score), 0.0), 1.0)

            created_at = getattr(episode, "created_at", None)
            if isinstance(created_at, datetime):
                safe_created_at = created_at if created_at.tzinfo is not None else created_at.replace(
                    tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - safe_created_at).total_seconds() / 86400.0)
            else:
                age_days = 365.0
            recency_score = math.exp(-age_days / 30.0)

            success_bonus = 0.15 if bool(episode.success) else -0.1
            access_count = max(int(getattr(episode, "access_count", 0) or 0), 0)
            reuse_bonus = min(math.log1p(access_count) * 0.05, 0.15)

            return (
                    safe_semantic * 0.65
                    + safe_episode_score * 0.2
                    + recency_score * 0.1
                    + reuse_bonus
                    + success_bonus
            )
        except (TypeError, ValueError, ArithmeticError):
            return 0.0

    @staticmethod
    def _episode_to_record(episode: Episode, actions: List[Action]) -> EpisodicMemoryRecord:
        action_items: List[Dict[str, Any]] = []
        if actions:
            action_items = [
                {
                    "step": action.step,
                    "tool_name": action.tool_name,
                    "tool_input": action.tool_input,
                    "tool_output": action.tool_output,
                }
                for action in actions
            ]

        return EpisodicMemoryRecord(
            record_id=episode.episode_id,
            importance=episode.importance,
            created_at=episode.created_at,
            last_accessed_at=episode.last_accessed_at or episode.created_at,
            session_id=episode.session_id,
            query=episode.query,
            result=episode.result,
            success=episode.success,
            score=episode.score,
            reflection=episode.reflection,
            access_count=episode.access_count,
            actions=action_items,
            tags=episode.tags or [],
        )
