from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from codecraft.llm.base import LLMProtocolError, LLMProvider, ModelRequest
from codecraft.llm.events import (
    ModelCompletedEvent,
    ModelEvent,
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
        """切分并校验预设事件脚本。

        Args:
            script: 使用 ``ModelCompletedEvent`` 划分模型调用边界的事件列表。

        Raises:
            ValueError: 某段响应缺少或重复 completed。

        Example:
            >>> from codecraft.llm.events import ModelMessageDeltaEvent
            >>> provider = MockProvider(
            ...     [
            ...         ModelMessageDeltaEvent(payload={"text": "done"}),
            ...         ModelCompletedEvent(),
            ...     ]
            ... )
            >>> len(provider._responses)
            1
        """
        self._responses = self._partition(script or [])
        self.calls: list[ModelRequest] = []
        self._lock = asyncio.Lock()

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        """原子认领下一段脚本响应，并记录请求快照后产出事件。

        Args:
            request: 当前 Runtime 发出的模型请求。

        Yields:
            下一段预设响应中的事件。

        Raises:
            LLMProtocolError: 脚本已经没有可用于本次调用的响应。

        Example:
            >>> async def collect_mock():
            ...     provider = MockProvider([ModelCompletedEvent()])
            ...     request = ModelRequest(model="test", messages=())
            ...     return [event.type async for event in provider.stream(request)]
            >>> asyncio.run(collect_mock())
            ['completed']
        """
        async with self._lock:
            if not self._responses:
                raise LLMProtocolError("mock provider has no response for this call")
            response = self._responses.pop(0)
            self.calls.append(request.model_copy(deep=True))

        for event in response:
            yield event

    @staticmethod
    def _partition(script: list[ModelEvent]) -> list[tuple[ModelEvent, ...]]:
        """按 completed 终止标志把平坦脚本切成多次模型响应。

        Args:
            script: 一个或多个响应顺序拼接的事件列表。

        Returns:
            每次响应一个不可变元组的列表。

        Raises:
            ValueError: 最后一段没有 completed，或某段响应内部结构非法。

        Example:
            >>> responses = MockProvider._partition(
            ...     [ModelCompletedEvent(), ModelCompletedEvent()]
            ... )
            >>> [len(response) for response in responses]
            [1, 1]
        """
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
        """校验一段 Mock 响应具有唯一且位于末尾的终止标志。

        Args:
            response: 已按 completed 边界切出的单次响应。

        Returns:
            ``None``。

        Raises:
            ValueError: completed 缺失、重复或不在末尾。

        Example:
            >>> MockProvider._validate_response((ModelCompletedEvent(),))
        """
        if not response or not isinstance(response[-1], ModelCompletedEvent):
            raise ValueError("each mock response must end with completed")
        if sum(isinstance(event, ModelCompletedEvent) for event in response) != 1:
            raise ValueError("each mock response must contain one completed event")
