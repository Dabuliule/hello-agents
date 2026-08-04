from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from codecraft.llm.base import LLMProtocolError, LLMProvider, ModelRequest
from codecraft.llm.events import (
    ModelCompletedEvent,
    ModelEvent,
    ModelMessageCompletedEvent,
    ModelMessageDeltaEvent,
)


class MockProvider(LLMProvider):
    """按模型调用边界返回预设事件的严格测试 Provider。

    ``script`` 使用显式 ``COMPLETED`` 划分每次调用。初始化时会先切分并校验
    所有响应；Mock 不补终止事件，避免测试脚本掩盖运行时或协议错误。
    """

    name = "mock"

    def __init__(
        self,
        script: list[ModelEvent] | None = None,
    ) -> None:
        self._responses = self._partition(script or [])
        self.calls: list[ModelRequest] = []
        self._lock = asyncio.Lock()

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        async with self._lock:
            if not self._responses:
                raise LLMProtocolError("mock provider has no response for this call")
            response = self._responses.pop(0)
            self.calls.append(request.model_copy(deep=True))

        for event in response:
            yield event

    @staticmethod
    def _partition(script: list[ModelEvent]) -> list[tuple[ModelEvent, ...]]:
        responses: list[tuple[ModelEvent, ...]] = []
        pending: list[ModelEvent] = []
        for event in script:
            pending.append(event)
            if isinstance(event, ModelCompletedEvent):
                response = tuple(pending)
                MockProvider._validate_response(response)
                responses.append(response)
                pending = []
        if pending:
            raise ValueError("mock script ended without completed")
        return responses

    @staticmethod
    def _validate_response(response: tuple[ModelEvent, ...]) -> None:
        if not response or not isinstance(response[-1], ModelCompletedEvent):
            raise ValueError("each mock response must end with completed")
        if sum(isinstance(event, ModelCompletedEvent) for event in response) != 1:
            raise ValueError("each mock response must contain one completed event")

        if any(isinstance(event, ModelMessageDeltaEvent) for event in response) and any(
            isinstance(event, ModelMessageCompletedEvent) for event in response
        ):
            raise ValueError(
                "a mock response cannot mix message deltas and a completed message"
            )
