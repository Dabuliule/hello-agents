from __future__ import annotations

from abc import ABC, abstractmethod

from codecraft.retrieval.models import RetrievalRequest, RetrievalResponse


class Retriever(ABC):
    """把一种检索实现约束为异步、标准请求/响应的策略接口。"""

    name: str

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """执行检索；暂时不可服务时应抛 RetrievalUnavailableError。"""
        ...
