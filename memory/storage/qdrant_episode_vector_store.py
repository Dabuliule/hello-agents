"""Qdrant 向量存储封装。"""

from __future__ import annotations

import hashlib
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


class QdrantEpisodeVectorStore:
    """仅负责写入 episode_id 与 embedding。"""

    def __init__(
            self,
            url: str,
            api_key: str | None = None,
            collection_name: str = "episodic_memory",
            vector_size: int = 128,
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection()

    def upsert(self, episode_id: str, embedding: List[float]) -> None:
        if len(embedding) != self._vector_size:
            raise ValueError(
                f"embedding 维度不匹配: 期望 {self._vector_size}, 实际 {len(embedding)}"
            )

        point_id = self._to_point_id(episode_id)
        point = qdrant_models.PointStruct(
            id=point_id,
            vector=embedding,
            payload={"episode_id": episode_id},
        )
        self._client.upsert(
            collection_name=self._collection_name,
            points=[point],
            wait=True,
        )

    def _ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection_name):
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _to_point_id(episode_id: str) -> int:
        # 使用稳定哈希生成 63 位正整数，满足 Qdrant 的整型 id 要求。
        digest = hashlib.sha256(episode_id.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return value & ((1 << 63) - 1)
