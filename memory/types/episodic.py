"""EpisodicMemory 实现。"""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from memory.base import MemoryBase, MemoryRecord
from memory.storage import Action, Episode, EpisodeNotFoundError, PostgresEpisodeStore


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

    def add(
            self,
            content: str,
            importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        metadata_value = metadata or {}
        episode = self._build_episode(content=content, importance=importance, metadata=metadata_value)
        actions = self._build_actions(metadata_value.get("actions"))
        self._store.insert_full_episode(episode=episode, actions=actions)
        return self._episode_to_record(episode=episode, actions=actions)

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

    def retrieve(
            self,
            query: Optional[str] = None,
            limit: int = 10,
            half_life_seconds: float = 86400.0,
    ) -> List[MemoryRecord]:
        if limit <= 0:
            return []

        candidates = self._store.query_episodes(limit=max(self._retrieve_window, limit * 5))
        scored: List[tuple[float, Episode]] = []
        for episode in candidates:
            score = self._retrieve_score(episode=episode, query=query, half_life_seconds=half_life_seconds)
            if score > 0.0:
                scored.append((score, episode))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_episodes = [episode for _, episode in scored[:limit]]
        return [self._episode_to_record(episode=ep, actions=[]) for ep in top_episodes]

    def delete(self, record_id: str) -> bool:
        return self._store.delete_episode(record_id)

    def clear(self) -> None:
        self._store.clear_episodes()

    def add_episode(self, episode: Episode, actions: Optional[List[Action]] = None) -> str:
        action_list = actions or []
        self._store.insert_full_episode(episode=episode, actions=action_list)
        return episode.episode_id

    def _build_episode(self, content: str, importance: float, metadata: Dict[str, Any]) -> Episode:
        tz = ZoneInfo("Asia/Shanghai")
        created_at = metadata.get("created_at")
        if isinstance(created_at, datetime):
            episode_created_at = created_at
        else:
            episode_created_at = datetime.now(tz)

        return Episode(
            episode_id=str(metadata.get("episode_id") or f"ep_{uuid.uuid4().hex}"),
            session_id=str(metadata.get("session_id") or self._default_session_id),
            query=content,
            result=str(metadata.get("result") or ""),
            success=bool(metadata.get("success", True)),
            score=float(metadata.get("score", 0.0)),
            reflection=self._normalize_optional_str(metadata.get("reflection")),
            importance=float(importance),
            created_at=episode_created_at,
            last_accessed_at=metadata.get("last_accessed_at") if isinstance(metadata.get("last_accessed_at"),
                                                                            datetime) else None,
            access_count=int(metadata.get("access_count", 0)),
            user_id=self._normalize_optional_str(metadata.get("user_id")),
            tags=self._normalize_tags(metadata.get("tags")),
        )

    @staticmethod
    def _build_actions(raw_actions: Any) -> List[Action]:
        if not isinstance(raw_actions, list):
            return []

        actions: List[Action] = []
        for index, item in enumerate(raw_actions, start=1):
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool_name") or item.get("tool") or "unknown"
            tool_input = item.get("tool_input")
            if not isinstance(tool_input, dict):
                tool_input = item.get("args") if isinstance(item.get("args"), dict) else {}
            tool_output = item.get("tool_output")
            if not isinstance(tool_output, dict):
                tool_output = {}

            actions.append(
                Action(
                    step=int(item.get("step", index)),
                    tool_name=str(tool_name),
                    tool_input=tool_input,
                    tool_output=tool_output,
                )
            )
        return actions

    @staticmethod
    def _episode_to_record(episode: Episode, actions: List[Action]) -> MemoryRecord:
        metadata: Dict[str, Any] = {
            "session_id": episode.session_id,
            "result": episode.result,
            "success": episode.success,
            "score": episode.score,
            "reflection": episode.reflection,
            "access_count": episode.access_count,
            "tags": episode.tags or [],
        }
        if actions:
            metadata["actions"] = [
                {
                    "step": action.step,
                    "tool_name": action.tool_name,
                    "tool_input": action.tool_input,
                    "tool_output": action.tool_output,
                }
                for action in actions
            ]

        return MemoryRecord(
            record_id=episode.episode_id,
            content=episode.query,
            importance=episode.importance,
            metadata=metadata,
            created_at=episode.created_at,
            last_accessed_at=episode.last_accessed_at or episode.created_at,
        )

    @staticmethod
    def _retrieve_score(episode: Episode, query: Optional[str], half_life_seconds: float) -> float:
        priority = EpisodicMemory._compute_priority(episode=episode, half_life_seconds=half_life_seconds)
        relevance = EpisodicMemory._keyword_score(text=episode.query, query=query)
        return 0.8 * priority + 0.2 * relevance

    @staticmethod
    def _compute_priority(episode: Episode, half_life_seconds: float) -> float:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds 必须大于 0")
        anchor = episode.last_accessed_at or episode.created_at
        now = datetime.now(anchor.tzinfo)
        age_seconds = max(0.0, (now - anchor).total_seconds())
        decay = math.exp(-math.log(2) * age_seconds / half_life_seconds)
        return max(float(episode.importance), 0.01) * decay

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []

        tokens: List[str] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()

        for ch in text.strip().lower():
            if ch.isspace() or ch in {
                ",",
                ".",
                "!",
                "?",
                ";",
                ":",
                "，",
                "。",
                "！",
                "？",
                "；",
                "：",
                "、",
                "（",
                "）",
                "(",
                ")",
                "[",
                "]",
                "{",
                "}",
                '"',
                "'",
                "`",
                "~",
                "@",
                "#",
                "$",
                "%",
                "^",
                "&",
                "*",
                "-",
                "_",
                "+",
                "=",
                "|",
                "\\",
                "/",
                "<",
                ">",
            }:
                flush_buffer()
                continue
            if "\u4e00" <= ch <= "\u9fff":
                flush_buffer()
                tokens.append(ch)
                continue
            if ch.isascii() and ch.isalnum():
                buffer.append(ch)
                continue
            flush_buffer()

        flush_buffer()
        return tokens

    @staticmethod
    def _keyword_score(text: str, query: Optional[str]) -> float:
        if not query:
            return 0.0
        normalized_query = query.strip().lower()
        if not normalized_query:
            return 0.0

        normalized_text = (text or "").strip().lower()
        if normalized_query in normalized_text:
            return 1.0

        text_tokens = set(EpisodicMemory._tokenize(normalized_text))
        query_tokens = EpisodicMemory._tokenize(normalized_query)
        if not text_tokens or not query_tokens:
            return 0.0

        overlap = sum(1 for token in query_tokens if token in text_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def _normalize_tags(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    @staticmethod
    def _normalize_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None
