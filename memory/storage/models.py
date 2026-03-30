"""PostgreSQL 情景记忆存储模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    episode_id: str
    session_id: str
    query: str
    result: str
    success: bool
    score: float
    reflection: Optional[str]
    importance: float = 0.5
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    # 预留：未来支持多租户。
    user_id: Optional[str] = None
    # 检索/过滤标签。
    tags: Optional[List[str]] = None


@dataclass
class Action:
    step: int
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
