"""WorkingMemory 实现。"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from memory.base import MemoryBase, MemoryRecord


class WorkingMemory(MemoryBase):
    """短期工作记忆：返回“最值得进入上下文”的 Top-K 记忆。"""

    def __init__(self, capacity: int = 500, ttl_seconds: int = 900):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds 必须大于 0")

        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        # 简化存储：WorkingMemory 只维护当前会话内存，不维护复杂索引。
        self._records: Dict[str, MemoryRecord] = {}

    def add(
            self,
            content: str,
            importance: float = 0.5,
            metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        self._cleanup_expired()
        record_id = f"wm_{uuid.uuid4().hex}"
        record = MemoryRecord(
            record_id=record_id,
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        self._records[record_id] = record
        self._enforce_capacity()
        return record

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        self._cleanup_expired()
        record = self._records.get(record_id)
        if not record:
            return None
        if self._is_expired(record):
            self._delete_record(record_id)
            return None
        # 只有 get 视为“真实访问”，会刷新 last_accessed_at。
        return self._touch_access(record_id, record)

    def list(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        self._cleanup_expired()
        records = list(self._records.values())
        # 按最近写入时间展示，保持可读性。
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit] if limit else records

    def retrieve(
            self,
            query: Optional[str] = None,
            limit: int = 10,
            half_life_seconds: float = 450.0,
    ) -> List[MemoryRecord]:
        self._cleanup_expired()
        if limit < 1:
            return []

        scored: List[tuple[float, MemoryRecord]] = []
        for record in self._records.values():
            score = self._retrieve_score(record, query=query, half_life_seconds=half_life_seconds)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def delete(self, record_id: str) -> bool:
        existed = record_id in self._records
        if existed:
            self._delete_record(record_id)
        return existed

    def clear(self) -> None:
        self._records.clear()

    def _retrieve_score(
            self,
            record: MemoryRecord,
            query: Optional[str],
            half_life_seconds: float,
    ) -> float:
        """统一评分：主信号 priority + 弱信号关键词相关性。"""
        priority = self.compute_priority(record, half_life_seconds=half_life_seconds)
        relevance = self._keyword_score(record.content, query)
        return 0.8 * priority + 0.2 * relevance

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """轻量中英文通用分词。

        - 英文/数字按连续片段组成 token
        - 中文按单字符切分
        - 忽略空白与常见分隔符
        """
        if not text:
            return []

        tokens: List[str] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            if buffer:
                tokens.append("".join(buffer))
                buffer.clear()

        for ch in text.strip().lower():
            if ch.isspace() or ch in {",", ".", "!", "?", ";", ":", "，", "。", "！", "？", "；", "：", "、", "（", "）", "(",
                                      ")", "[", "]", "{", "}", "\"", "'", "`", "~", "@", "#", "$", "%", "^", "&", "*",
                                      "-", "_", "+", "=", "|", "\\", "/", "<", ">"}:
                flush_buffer()
                continue

            # 中文字符按单字切分
            if "\u4e00" <= ch <= "\u9fff":
                flush_buffer()
                tokens.append(ch)
                continue

            # 英文/数字继续拼接
            if ch.isascii() and ch.isalnum():
                buffer.append(ch)
                continue

            # 其他字符作为分隔处理
            flush_buffer()

        flush_buffer()
        return tokens

    def _keyword_score(self, text: str, query: Optional[str]) -> float:
        """轻量关键词相关性：substring 强匹配 + token overlap。"""
        if not query:
            return 0.0

        normalized_query = query.strip().lower()
        if not normalized_query:
            return 0.0

        normalized_text = (text or "").strip().lower()

        # 强匹配优先：子串命中直接最高分
        if normalized_query in normalized_text:
            return 1.0

        text_tokens = set(self._tokenize(normalized_text))
        query_tokens = self._tokenize(normalized_query)
        if not text_tokens or not query_tokens:
            return 0.0

        overlap = sum(1 for token in query_tokens if token in text_tokens)
        return overlap / len(query_tokens)

    @staticmethod
    def compute_priority(
            record: MemoryRecord,
            now: Optional[datetime] = None,
            half_life_seconds: float = 450.0,
    ) -> float:
        """priority = importance * exp_decay（基于 last_accessed_at）。"""
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds 必须大于 0")

        anchor = record.last_accessed_at or record.created_at
        current = now or datetime.now(anchor.tzinfo)
        age_seconds = max(0.0, (current - anchor).total_seconds())
        decay = math.exp(-math.log(2) * age_seconds / half_life_seconds)

        effective_importance = max(float(record.importance), 0.01)
        return effective_importance * decay

    def _touch_access(self, record_id: str, record: MemoryRecord) -> MemoryRecord:
        now = datetime.now(record.created_at.tzinfo)
        if record.last_accessed_at and record.last_accessed_at >= now:
            return record

        updated = MemoryRecord(
            record_id=record.record_id,
            content=record.content,
            importance=record.importance,
            metadata=record.metadata,
            created_at=record.created_at,
            last_accessed_at=now,
        )
        self._records[record_id] = updated
        return updated

    def _cleanup_expired(self) -> None:
        expired_ids = [rid for rid, rec in self._records.items() if self._is_expired(rec)]
        for record_id in expired_ids:
            self._delete_record(record_id)

    def _enforce_capacity(self) -> None:
        # 软淘汰：先清理 TTL；硬淘汰：容量超限时按实时 priority 删除最低者。
        while len(self._records) > self.capacity:
            victim_id = self._select_eviction_candidate()
            if victim_id is None:
                break
            self._delete_record(victim_id)

    def _select_eviction_candidate(self) -> Optional[str]:
        if not self._records:
            return None

        # 使用实时 priority 计算，保证淘汰决策与当前时间一致。
        min_id: Optional[str] = None
        min_priority: Optional[float] = None
        oldest_anchor: Optional[datetime] = None

        for record_id, record in self._records.items():
            priority = self.compute_priority(record)
            anchor = record.last_accessed_at or record.created_at
            if min_priority is None or priority < min_priority:
                min_priority = priority
                min_id = record_id
                oldest_anchor = anchor
                continue
            if priority == min_priority and oldest_anchor is not None and anchor < oldest_anchor:
                min_id = record_id
                oldest_anchor = anchor

        return min_id

    def _delete_record(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def _is_expired(self, record: MemoryRecord) -> bool:
        anchor = record.last_accessed_at or record.created_at
        return datetime.now(anchor.tzinfo) >= anchor + timedelta(seconds=self.ttl_seconds)
