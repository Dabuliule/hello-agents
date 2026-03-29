"""Memory 模块基类。"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MemoryRecord:
    """基础记忆条目：仅包含通用字段。"""

    record_id: str
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Shanghai")))
    last_accessed_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id 不能为空")
        if not isinstance(self.importance, (int, float)):
            raise ValueError("importance 必须是数字")
        if not math.isfinite(float(self.importance)):
            raise ValueError("importance 必须是有限数值")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at 必须包含时区信息")
        if self.last_accessed_at is None:
            object.__setattr__(self, "last_accessed_at", self.created_at)
        if self.last_accessed_at.tzinfo is None:
            raise ValueError("last_accessed_at 必须包含时区信息")


class MemoryBase(ABC):
    """记忆基类，定义统一接口。"""

    @abstractmethod
    def add(self, content: str, importance: float = 0.0, metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """新增一条记忆并返回记录。"""

    def add_many(self, items: Iterable[tuple[str, float, Optional[Dict[str, Any]]]]) -> List[MemoryRecord]:
        """批量新增记忆。"""
        records: List[MemoryRecord] = []
        for content, importance, metadata in items:
            records.append(self.add(content, importance, metadata))
        return records

    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        """按 ID 获取记忆。"""

    @abstractmethod
    def list(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        """列出记忆。"""

    @abstractmethod
    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryRecord]:
        """按统一评分返回 Top-K 记忆。"""

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """删除记忆，返回是否删除成功。"""

    @abstractmethod
    def clear(self) -> None:
        """清空全部记忆。"""
