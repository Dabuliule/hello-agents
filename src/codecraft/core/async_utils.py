from __future__ import annotations

import asyncio
from typing import TypeVar


_T = TypeVar("_T")


async def finish_task_before_cancelling(task: asyncio.Task[_T]) -> _T:
    """先收口已经启动的不可撤销副作用，再传播调用者取消。

    ``shield`` 只阻止当前等待者把取消传给 ``task``，不会吞掉等待者自身的
    ``CancelledError``。若等待期间又收到取消，就记住最后一次取消并继续 shield，直到
    task 真正结束；随后先用 ``task.result()`` 暴露副作用自身的失败，否则再重新抛出
    取消。它适用于文件 append、容器清理等“已经开始就必须知道最终结果”的操作。

    Args:
        task: 已经通过 ``create_task`` 启动的单个副作用。

    Returns:
        task 正常完成时的结果。

    Raises:
        asyncio.CancelledError: task 收口成功后，重新传播等待者收到的取消。
        Exception: task 自身失败时优先传播该失败，避免把真实副作用结果伪装成取消。
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc

    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result
