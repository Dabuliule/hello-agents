"""WorkingMemory 实现。"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from memory.base import MemoryBase, MemoryRecord


class WorkingMemory(MemoryBase):
    """纯内存工作记忆，支持自动清理。"""

    def __init__(
            self,
            capacity: int = 500,
            ttl_seconds: int = 900,
    ):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds 必须大于 0")

        self.capacity = capacity
        self.ttl_seconds = ttl_seconds

        # 使用 OrderedDict 维护 LRU 顺序（右侧为最新访问）。
        self._records: "OrderedDict[str, MemoryRecord]" = OrderedDict()

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
        self._records.move_to_end(record_id)
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
        return self._touch_access(record_id, record)

    def list(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        self._cleanup_expired()
        records: List[MemoryRecord] = []
        for record in reversed(self._records.values()):
            if self._is_expired(record):
                self._delete_record(record.record_id)
                continue
            # list/search 不更新访问时间，避免“读取污染”
            records.append(record)
            if limit and len(records) >= limit:
                break
        return records

    def search(
            self,
            query: str,
            limit: Optional[int] = None,
            sort_by_priority: bool = False,
            half_life_seconds: float = 450.0,
    ) -> List[MemoryRecord]:
        self._cleanup_expired()
        normalized = query.strip().lower()
        if not normalized:
            return []
        results: List[MemoryRecord] = []
        scored: List[tuple[float, MemoryRecord]] = []

        for record in reversed(self._records.values()):
            if self._is_expired(record):
                self._delete_record(record.record_id)
                continue
            if normalized in record.content.lower():
                if sort_by_priority:
                    score = self._score(record, half_life_seconds=half_life_seconds)
                    scored.append((score, record))
                else:
                    results.append(record)
                    if limit and len(results) >= limit:
                        break

        if sort_by_priority:
            scored.sort(key=lambda item: item[0], reverse=True)
            results = [record for _, record in scored[:limit]] if limit else [record for _, record in scored]

        return results

    def delete(self, record_id: str) -> bool:
        """删除记忆，返回是否删除成功。"""
        existed = record_id in self._records
        if existed:
            self._delete_record(record_id)
        return existed

    def clear(self) -> None:
        """清空全部记忆。"""
        self._records.clear()

    def _score(self, record: MemoryRecord, half_life_seconds: float) -> float:
        """优先级打分钩子，便于未来扩展 relevance/embedding。"""
        return self.compute_priority(record, half_life_seconds=half_life_seconds)

    @staticmethod
    def compute_priority(
            record: MemoryRecord,
            now: Optional[datetime] = None,
            half_life_seconds: float = 450.0,
    ) -> float:
        """计算优先级 = 重要性 * 时间衰减因子。"""
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds 必须大于 0")
        current = now or datetime.now(record.created_at.tzinfo)
        anchor = record.last_accessed_at or record.created_at
        age_seconds = max(0.0, (current - anchor).total_seconds())
        decay = math.exp(-math.log(2) * age_seconds / half_life_seconds)
        # importance 设定下限，避免优先级长期为 0
        effective_importance = max(float(record.importance), 0.01)
        return effective_importance * decay

    def _touch_access(self, record_id: str, record: MemoryRecord) -> MemoryRecord:
        # 仅 get 更新访问时间，且会移动 LRU 顺序
        now = datetime.now(record.created_at.tzinfo)
        if record.last_accessed_at and record.last_accessed_at >= now:
            self._records.move_to_end(record_id)
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
        self._records.move_to_end(record_id)
        return updated

    def _cleanup_expired(self) -> None:
        expired_ids = [rid for rid, rec in list(self._records.items()) if self._is_expired(rec)]
        for record_id in expired_ids:
            self._delete_record(record_id)

    def _enforce_capacity(self) -> None:
        # 软淘汰：优先清理 TTL 过期；硬淘汰：容量超限时按实时优先级淘汰
        while len(self._records) > self.capacity:
            victim_id = self._select_eviction_candidate()
            if victim_id is None:
                break
            self._delete_record(victim_id)

    def _delete_record(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def _select_eviction_candidate(self) -> Optional[str]:
        """容量淘汰时按实时优先级选择最低者。"""
        if not self._records:
            return None
        min_id: Optional[str] = None
        min_score: Optional[float] = None
        for record_id, record in self._records.items():
            score = self._score(record, half_life_seconds=450.0)
            if min_score is None or score < min_score:
                min_score = score
                min_id = record_id
        return min_id

    def _is_expired(self, record: MemoryRecord) -> bool:
        anchor = record.last_accessed_at or record.created_at
        expires_at = anchor + timedelta(seconds=self.ttl_seconds)
        return datetime.now(record.created_at.tzinfo) >= expires_at
