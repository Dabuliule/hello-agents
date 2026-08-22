from __future__ import annotations

from dataclasses import dataclass, replace

from codecraft.core.conversation import (
    Conversation,
    ConversationToolCallItem,
    ConversationToolResultItem,
)
from codecraft.schema.event import RuntimeEvent, RuntimeEventType
from codecraft.schema.tool import ToolResult


@dataclass(frozen=True, slots=True)
class IncompleteToolCall:
    """日志中有模型调用意图、但没有持久化终态结果的工具调用。"""

    call_id: str
    name: str
    turn_id: str | None
    started: bool


@dataclass(frozen=True, slots=True)
class ReconstructedSessionState:
    """已加载事件分析得到的模型历史和待保守收口工具调用。"""

    conversation: Conversation
    incomplete_tool_calls: tuple[IncompleteToolCall, ...]


def reconstruct_conversation(events: list[RuntimeEvent]) -> Conversation:
    """把已验证事件投影为下一次模型请求所需的 Conversation。

    Args:
        events: 已由 SessionStore 校验 schema、session_id 和连续 seq 的完整日志。

    Returns:
        按事件顺序重建的模型可见历史。新建 item 的内部 ID/时间可以不同，但
        ``build_model_messages`` 的角色、文本、tool call 和 result 语义保持一致。

    Raises:
        ValueError: ``CONTEXT_COMPACTED`` 缺少有效的完整 Conversation 快照。

    这是纯状态投影，不是事件重放：不会调用 LLM、执行工具、重新审批或广播旧事件。
    只消费影响模型协议的 USER_MESSAGE、完整 ASSISTANT_MESSAGE、MODEL_TOOL_CALL、
    TOOL_CALL_FINISHED 和 CONTEXT_COMPACTED。Delta、生命周期、审批、Token 和诊断事件
    都不会进入模型历史。

    CONTEXT_COMPACTED 携带压缩后的完整快照，因此遇到它时替换此前投影，再继续消费
    后续事件。协作式 interrupt/timeout 会在退出前为已记录 calls 补齐失败结果；但
    进程硬崩溃可能留下未配对的 MODEL_TOOL_CALL；本函数仍保持纯投影、不合成结果，
    AgentRuntime 会使用 ``reconstruct_session_state`` 的分析结果追加显式恢复终态。
    """
    conversation = Conversation()

    for event in events:
        if event.type == RuntimeEventType.USER_MESSAGE:
            conversation.append_user_message(str(event.payload.get("text", "")))

        elif event.type == RuntimeEventType.ASSISTANT_MESSAGE:
            conversation.append_assistant_message(str(event.payload.get("text", "")))

        elif event.type == RuntimeEventType.MODEL_TOOL_CALL:
            call_id = str(event.payload.get("call_id", ""))
            name = str(event.payload.get("name", ""))
            arguments = event.payload.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            conversation.append_model_tool_call(call_id, name, arguments)

        elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
            call_id = str(event.payload.get("call_id", ""))
            name = str(event.payload.get("name", ""))
            result = event.payload.get("result")
            content = ""
            if isinstance(result, dict):
                content = ToolResult.model_validate(result).model_content()
            conversation.append_tool_result(call_id, name, content)

        elif event.type == RuntimeEventType.CONTEXT_COMPACTED:
            snapshot = event.payload.get("conversation")
            if not isinstance(snapshot, dict):
                raise ValueError("context_compacted event is missing its conversation")
            # 快照已经包含压缩时保留的近期历史；若继续叠加此前事件，会把被摘要替代的
            # 旧消息重新放回模型上下文。替换后，循环仍会追加 compaction 之后的新事件。
            conversation = Conversation.model_validate(snapshot)

    return conversation


def reconstruct_session_state(events: list[RuntimeEvent]) -> ReconstructedSessionState:
    """组合 Conversation 投影与工具生命周期分析，供 Resume 安全续接。

    事件已经在内存中，两个职责各自线性扫描一次；这避免把模型消息投影和崩溃协议
    修复揉进一个高复杂度状态机，磁盘 JSONL 仍只由 SessionStore 读取一遍。

    ``MODEL_TOOL_CALL`` 建立 pending 项，``TOOL_CALL_STARTED`` 只把它升级为可能已产生
    副作用，``TOOL_CALL_FINISHED`` 才关闭协议。结果按模型调用出现顺序返回，使 Resume
    追加的保守 ToolResult 顺序稳定。

    ``CONTEXT_COMPACTED`` 会替换 Conversation，因此 pending 集合也必须从快照重建；
    快照若含没有结果的 call，已经无法证明它从未启动，按最保守的 started=True 处理。
    """
    return ReconstructedSessionState(
        conversation=reconstruct_conversation(events),
        incomplete_tool_calls=_find_incomplete_tool_calls(events),
    )


def _find_incomplete_tool_calls(
    events: list[RuntimeEvent],
) -> tuple[IncompleteToolCall, ...]:
    """按 call_id 分析工具生命周期，返回缺少终态的调用。

    ``MODEL_TOOL_CALL`` 建立 pending 项，``TOOL_CALL_STARTED`` 只把它升级为可能已产生
    副作用，``TOOL_CALL_FINISHED`` 才关闭协议。结果按模型调用出现顺序返回，使 Resume
    追加的保守 ToolResult 顺序稳定。

    ``CONTEXT_COMPACTED`` 会替换 Conversation，因此 pending 集合也必须从快照重建；
    快照若含没有结果的 call，已经无法证明它从未启动，按最保守的 started=True 处理。
    """
    pending: dict[str, IncompleteToolCall] = {}

    for event in events:
        if event.type == RuntimeEventType.MODEL_TOOL_CALL:
            call_id = str(event.payload.get("call_id", ""))
            name = str(event.payload.get("name", ""))
            pending[call_id] = IncompleteToolCall(
                call_id=call_id,
                name=name,
                turn_id=event.turn_id,
                started=False,
            )

        elif event.type == RuntimeEventType.TOOL_CALL_STARTED:
            call_id = str(event.payload.get("call_id", ""))
            incomplete = pending.get(call_id)
            if incomplete is not None:
                pending[call_id] = replace(incomplete, started=True)

        elif event.type == RuntimeEventType.TOOL_CALL_FINISHED:
            call_id = str(event.payload.get("call_id", ""))
            pending.pop(call_id, None)

        elif event.type == RuntimeEventType.CONTEXT_COMPACTED:
            snapshot = event.payload.get("conversation")
            if not isinstance(snapshot, dict):
                raise ValueError("context_compacted event is missing its conversation")
            pending = _pending_calls_from_conversation(
                Conversation.model_validate(snapshot)
            )

    return tuple(pending.values())


def _pending_calls_from_conversation(
    conversation: Conversation,
) -> dict[str, IncompleteToolCall]:
    """从压缩快照恢复未配对 calls，并对未知执行状态采用保守判断。"""
    pending: dict[str, IncompleteToolCall] = {}
    for item in conversation.items:
        if isinstance(item, ConversationToolCallItem):
            pending[item.tool_call_id] = IncompleteToolCall(
                call_id=item.tool_call_id,
                name=item.name,
                turn_id=None,
                started=True,
            )
        elif isinstance(item, ConversationToolResultItem):
            pending.pop(item.tool_call_id, None)
    return pending
