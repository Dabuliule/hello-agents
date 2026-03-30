"""嵌入服务。"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Optional


class EmbeddingService:
    """可替换的嵌入服务封装。"""

    def __init__(
            self,
            provider: Optional[Callable[[str], List[float]]] = None,
            dimension: int = 128,
    ) -> None:
        self._provider = provider
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        if self._provider is not None:
            return self._provider(text)
        return self._mock_embed(text)

    def _mock_embed(self, text: str) -> List[float]:
        # 使用稳定哈希生成可复现向量，便于本地联调。
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: List[float] = []
        for i in range(self._dimension):
            byte_value = seed[i % len(seed)]
            values.append((byte_value / 255.0) * 2.0 - 1.0)
        return values
