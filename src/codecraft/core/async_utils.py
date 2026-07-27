from __future__ import annotations

import asyncio
from typing import TypeVar


_T = TypeVar("_T")


async def finish_task_before_cancelling(task: asyncio.Task[_T]) -> _T:
    """Finish an in-flight side effect before propagating caller cancellation."""
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
