"""创建、恢复和关闭 Agent Thread 的顶层 Runtime 资源所有者。"""

from __future__ import annotations

from collections.abc import Sequence
import asyncio
from pathlib import Path
from typing import Any

from codecraft.approval.manager import ApprovalManager
from codecraft.core.event_bus import EventBus
from codecraft.core.reconstruction import reconstruct_conversation
from codecraft.core.session import Session
from codecraft.core.session_store import SessionStore
from codecraft.core.thread import AgentThread
from codecraft.llm.registry import LLMProviderRegistry
from codecraft.schema.event import RuntimeEventType
from codecraft.schema.session import SessionConfig, SessionSnapshot, SessionSummary
from codecraft.skill import SkillRegistry
from codecraft.tool.registry import ToolRegistry
from codecraft.tool.observer import ToolResultObserver


class AgentRuntime:
    """连接持久化、模型、工具和审批组件的应用层运行时入口。

    Runtime 不执行 Agent Loop，也不保存某个 Session 的 conversation。它负责
    验证 SessionConfig、启动共享 Tool Provider、选择模型 Provider、创建 Session
    与面向 UI 的 AgentThread，并在应用退出时关闭其拥有的异步资源。
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        llm_providers: LLMProviderRegistry,
        tool_registry: ToolRegistry,
        approval_manager: ApprovalManager | None = None,
        event_bus: EventBus | None = None,
        tool_result_observers: Sequence[ToolResultObserver] | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        """注入创建和恢复 Session 所需的共享依赖。

        Args:
            session_store: Session JSONL 日志的持久化边界。
            llm_providers: 名称到模型 Provider 的生命周期 Registry。
            tool_registry: 内置工具和动态 Tool Provider 的生命周期 Registry。
            approval_manager: 可选的共享审批评估器和 Reviewer。
            event_bus: 可选的外部事件总线；缺失时每个 Session 创建独立总线。
            tool_result_observers: 工具成功后执行的非关键后处理器。
            skill_registry: Prompt 目录和 ``load_skill`` 共享的 Skill 来源。

        构造函数只保存依赖，不启动 Provider、创建日志或调用模型。需要异步资源的
        初始化延迟到 ``create_thread``/``resume_snapshot``。
        """
        self.session_store = session_store
        self.llm_providers = llm_providers
        self.tool_registry = tool_registry
        self.approval_manager = approval_manager or ApprovalManager()
        self.event_bus = event_bus
        self.tool_result_observers = tuple(tool_result_observers or ())
        self.skill_registry = (
            skill_registry if skill_registry is not None else SkillRegistry()
        )

    async def create_thread(self, config: SessionConfig) -> AgentThread:
        """以确定的启动顺序创建一个全新 Session，并返回其 UI facade。

        Args:
            config: 已解析、带新 Session ID 的完整执行快照。

        Returns:
            已订阅 Session EventBus、队列中包含 ``SESSION_STARTED`` 的 AgentThread。

        Raises:
            ValueError: cwd 当前不可执行，或配置的 Provider 不存在。
            CodecraftError: Tool Provider 启动或 Session 日志创建失败。

        先验证 cwd、选择模型并完整启动 ToolRegistry，随后才创建日志和 Session。
        Thread 在首事件发出前订阅 EventBus，因此调用方拿到它时不会遗漏
        ``SESSION_STARTED``；任何启动失败都不会暴露一个部分初始化的 Thread。
        """
        config.ensure_runtime_ready()
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
            skill_registry=self.skill_registry,
        )
        thread = AgentThread(session)
        skill_snapshot = self._skill_snapshot()
        await session.emit(
            RuntimeEventType.SESSION_STARTED,
            {
                "config": config.model_dump(mode="json"),
                **({"skills": skill_snapshot} if skill_snapshot else {}),
            },
        )
        return thread

    async def resume_thread(self, session_id: str) -> AgentThread:
        """根据 session 日志恢复 thread，并重建模型 conversation。"""
        snapshot = await self.session_store.resume(session_id)
        return await self.resume_snapshot(snapshot)

    async def resume_snapshot(self, snapshot: SessionSnapshot) -> AgentThread:
        """从已加载的快照恢复 thread，避免重复读取同一份 session 日志。"""
        snapshot.config.ensure_runtime_ready()
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
            skill_registry=self.skill_registry,
            conversation=conversation,
            seq=snapshot.events[-1].seq if snapshot.events else 0,
        )
        thread = AgentThread(session)
        skill_snapshot = self._skill_snapshot()
        await session.emit(
            RuntimeEventType.SESSION_RESTORED,
            {"skills": skill_snapshot} if skill_snapshot else None,
        )
        return thread

    async def resume_last(self, cwd: Path | None = None) -> AgentThread:
        """恢复可选 cwd 范围内最近修改且有效的 Session。"""
        snapshot = await self.session_store.resume_last(cwd=cwd)
        return await self.resume_snapshot(snapshot)

    async def list_sessions(self, cwd: Path | None = None) -> list[SessionSummary]:
        """列出可选 cwd 范围内的可恢复 Session 摘要。"""
        return await self.session_store.list_sessions(cwd=cwd)

    async def close(self) -> None:
        """并发关闭模型和工具资源，并保证两组资源都获得清理机会。

        Raises:
            RuntimeError: 至少一组资源关闭失败；首个异常作为 cause 保留。

        ``return_exceptions=True`` 防止一边关闭失败后跳过另一边的清理。Runtime
        只关闭自己拥有的 Registry；具体 Session 应由对应 AgentThread 先关闭。
        """
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

    def _skill_snapshot(self) -> dict[str, list[dict[str, Any]]] | None:
        """序列化可用 Skill 与发现诊断；二者都空时省略事件字段。"""
        available = [
            metadata.model_dump(mode="json") for metadata in self.skill_registry.list()
        ]
        diagnostics = [
            diagnostic.model_dump(mode="json")
            for diagnostic in self.skill_registry.diagnostics()
        ]
        if not available and not diagnostics:
            return None
        return {"available": available, "diagnostics": diagnostics}
