"""在 Session 内按顺序发布 RuntimeEvent 的轻量异步总线。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable

from codecraft.schema.event import RuntimeEvent

EventHandler = Callable[[RuntimeEvent], Awaitable[None]]


class EventBus:
    """RuntimeEvent 的轻量异步事件总线。

    handler 按订阅顺序串行执行；异常会继续抛给调用方，避免事件投影失败后被静默
    吞掉。它不负责持久化、并发 fan-out 或事件重放；Session.emit 在调用总线前
    已经把事件写入 SessionStore。
    """

    def __init__(self) -> None:
        """创建按登记顺序保存 handler 的空事件总线。"""
        self._handlers: list[EventHandler] = []

    def subscribe(
        self,
        handler: EventHandler,
    ) -> None:
        """按登记顺序注册一个异步事件处理器。

        Args:
            handler: 接收单个 RuntimeEvent 并返回 Awaitable 的 callable。

        handler 的异步形态在首次 emit 时验证，以允许普通函数返回自定义
        Awaitable，同时拒绝完全同步的事件副作用。
        """
        self._handlers.append(handler)

    async def emit(
        self,
        event: RuntimeEvent,
    ) -> None:
        """对订阅列表快照按顺序发送一个事件。

        Args:
            event: 已由 Session 分配 seq 并持久化的 RuntimeEvent。

        Raises:
            TypeError: handler 返回值不可等待。
            Exception: 任一 handler 失败时原样传播，后续 handler 不再执行。

        遍历列表快照使 handler 在回调期间新增订阅者时，不会让新订阅者收到正在
        发布的半途事件。串行 await 则保持所有观察者看到一致的事件顺序。
        """
        for handler in list(self._handlers):
            result = handler(event)

            if not isawaitable(result):
                raise TypeError("EventBus handler must be an async callable")

            await result
