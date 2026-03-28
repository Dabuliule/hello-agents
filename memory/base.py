"""Memory 模块基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MemoryRecord:
    """记忆条目。"""

    record_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(ZoneInfo("Asia/Shanghai")))


class MemoryBase(ABC):
    """记忆基类，定义统一接口。"""

    @abstractmethod
    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """新增一条记忆并返回记录。"""

    def add_many(self, items: Iterable[tuple[str, Optional[Dict[str, Any]]]]) -> List[MemoryRecord]:
        """批量新增记忆。"""
        records: List[MemoryRecord] = []
        for content, metadata in items:
            records.append(self.add(content, metadata))
        return records

    @abstractmethod
    def get(self, record_id: str) -> Optional[MemoryRecord]:
        """按 ID 获取记忆。"""

    @abstractmethod
    def list(self, limit: Optional[int] = None) -> List[MemoryRecord]:
        """列出记忆。"""

    @abstractmethod
    def search(self, query: str, limit: Optional[int] = None) -> List[MemoryRecord]:
        """搜索记忆。"""

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """删除记忆，返回是否删除成功。"""

    @abstractmethod
    def clear(self) -> None:
        """清空全部记忆。"""
