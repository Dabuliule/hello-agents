from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable

from codecraft.schema.event import RuntimeEvent

EventHandler = Callable[[RuntimeEvent], Awaitable[None]]


class EventBus:
    """RuntimeEvent 的轻量异步事件总线。

    handler 按订阅顺序执行；异常会继续抛给调用方，避免事件副作用失败后被静默
    吞掉。
    """

    def __init__(self) -> None:
        """创建按登记顺序保存 handler 的空事件总线。"""
        self._handlers: list[EventHandler] = []

    def subscribe(
        self,
        handler: EventHandler,
    ) -> None:
        """注册一个异步事件处理器。"""
        self._handlers.append(handler)

    async def emit(
        self,
        event: RuntimeEvent,
    ) -> None:
        """按订阅顺序发送事件。"""
        for handler in list(self._handlers):
            result = handler(event)

            if not isawaitable(result):
                raise TypeError("EventBus handler must be an async callable")

            await result
