from __future__ import annotations

from collections.abc import Sequence
import asyncio
from pathlib import Path

from codecraft.approval.manager import ApprovalManager
from codecraft.core.event_bus import EventBus
from codecraft.core.reconstruction import reconstruct_conversation
from codecraft.core.session import Session
from codecraft.core.session_store import SessionStore
from codecraft.core.thread import AgentThread
from codecraft.llm.registry import LLMProviderRegistry
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSummary
from codecraft.tool.registry import ToolRegistry
from codecraft.tool.observer import ToolResultObserver


class AgentRuntime:
    """装配 session store、LLM provider 和 tool registry 的运行时入口。"""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        llm_providers: LLMProviderRegistry,
        tool_registry: ToolRegistry,
        approval_manager: ApprovalManager | None = None,
        event_bus: EventBus | None = None,
        tool_result_observers: Sequence[ToolResultObserver] | None = None,
    ) -> None:
        self.session_store = session_store
        self.llm_providers = llm_providers
        self.tool_registry = tool_registry
        self.approval_manager = approval_manager or ApprovalManager()
        self.event_bus = event_bus
        self.tool_result_observers = tuple(tool_result_observers or ())

    async def create_thread(self, config: SessionConfig) -> AgentThread:
        """创建新 session，并返回可消费事件的 AgentThread。"""
        llm_provider = self.llm_providers.get(config.model_provider)
        await self.tool_registry.start()
        await self.session_store.create_session(config)
        session = Session(
            config=config,
            session_store=self.session_store,
            llm_provider=llm_provider,
            tool_registry=self.tool_registry,
            approval_manager=self.approval_manager,
            event_bus=self.event_bus,
            tool_result_observers=self.tool_result_observers,
        )
        thread = AgentThread(session)
        await session.emit(
            RuntimeEventType.SESSION_STARTED,
            {"config": config.model_dump(mode="json")},
        )
        return thread

    async def resume_thread(self, session_id: str) -> AgentThread:
        """根据 session 日志恢复 thread，并重建模型 conversation。"""
        snapshot = await self.session_store.resume(session_id)
        llm_provider = self.llm_providers.get(snapshot.config.model_provider)
        await self.tool_registry.start()
        conversation = reconstruct_conversation(snapshot.events)

        session = Session(
            config=snapshot.config,
            session_store=self.session_store,
            llm_provider=llm_provider,
            tool_registry=self.tool_registry,
            approval_manager=self.approval_manager,
            event_bus=self.event_bus,
            tool_result_observers=self.tool_result_observers,
            conversation=conversation,
            seq=snapshot.events[-1].seq if snapshot.events else 0,
        )
        thread = AgentThread(session)
        await session.emit(RuntimeEventType.SESSION_RESTORED)
        return thread

    async def resume_last(self, cwd: Path | None = None) -> AgentThread:
        snapshot = await self.session_store.resume_last(cwd=cwd)
        return await self.resume_thread(snapshot.config.session_id)

    async def list_sessions(self, cwd: Path | None = None) -> list[SessionSummary]:
        return await self.session_store.list_sessions(cwd=cwd)

    async def close(self) -> None:
        """关闭模型和工具资源，并保证两边都获得清理机会。"""
        results = await asyncio.gather(
            self.llm_providers.close(),
            self.tool_registry.close(),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(
                f"failed to close {len(errors)} runtime resource group(s)"
            ) from errors[0]
