from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from codecraft.retrieval.errors import RetrievalUnavailableError
from codecraft.retrieval.models import RetrievalRequest, RetrievalResponse
from codecraft.retrieval.retrievers import Retriever, ScanRetriever
from codecraft.retrieval.router import QueryRouter


class ContextEngine:
    """Retrieval boundary used by tools and future query routing."""

    def __init__(
        self,
        retrievers: Sequence[Retriever] | None = None,
        *,
        default_retriever: str | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        """注册唯一名称 Retriever，并选择默认实现和查询路由器。

        Raises:
            ValueError: Retriever 为空、名称重复或默认名称不存在。
        """
        configured = tuple(retrievers) if retrievers is not None else (ScanRetriever(),)
        if not configured:
            raise ValueError("context engine requires at least one retriever")
        names = [retriever.name for retriever in configured]
        if len(names) != len(set(names)):
            raise ValueError("context engine retrievers must have unique names")
        self._retrievers = {retriever.name: retriever for retriever in configured}
        self._default_retriever = default_retriever or names[0]
        if self._default_retriever not in self._retrievers:
            raise ValueError(f"unknown default retriever: {self._default_retriever}")
        self._router = router or QueryRouter()

    @property
    def retriever_names(self) -> tuple[str, ...]:
        """按配置顺序返回可选 Retriever 名称。"""
        return tuple(self._retrievers)

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        retriever_name: str | None = None,
        fallback_retriever: str | None = None,
    ) -> RetrievalResponse:
        """通过指定、默认或 auto 策略检索，并记录完整尝试链。

        Args:
            request: 查询及 workspace 约束。
            retriever_name: 实现名；``auto`` 表示使用 QueryRouter。
            fallback_retriever: 首选未知或不可用时的显式备用实现。

        Returns:
            补齐 retriever、fallback_from、route_reason 和 attempted_retrievers
            的标准响应。

        Raises:
            ValueError: 指定的实现名未知且没有可用 fallback。
            RetrievalUnavailableError: 实现存在但当前无法服务，且没有成功降级。
        """
        selected = retriever_name or self._default_retriever
        if selected == "auto":
            return await self._retrieve_auto(request)
        try:
            retriever = self._retrievers[selected]
        except KeyError as exc:
            if fallback_retriever is None:
                raise ValueError(f"unknown retriever: {selected}") from exc
            return await self._fallback(request, selected, fallback_retriever)
        try:
            response = await retriever.retrieve(request)
        except RetrievalUnavailableError:
            if fallback_retriever is None or fallback_retriever == selected:
                raise
            return await self._fallback(request, selected, fallback_retriever)
        return replace(
            response,
            retriever=selected,
            attempted_retrievers=(selected,),
        )

    async def _retrieve_auto(self, request: RetrievalRequest) -> RetrievalResponse:
        """顺序执行路由计划，遇到首个非空结果即停止。"""
        plan = self._router.route(request)
        attempted: list[str] = []
        last_response: RetrievalResponse | None = None
        for name in plan.retrievers:
            retriever = self._retrievers.get(name)
            if retriever is None:
                continue
            attempted.append(name)
            try:
                response = await retriever.retrieve(request)
            except RetrievalUnavailableError:
                continue
            last_response = response
            if response.matches:
                return replace(
                    response,
                    retriever=name,
                    route_reason=plan.reason,
                    attempted_retrievers=tuple(attempted),
                )

        if last_response is not None:
            return replace(
                last_response,
                retriever=attempted[-1],
                route_reason=plan.reason,
                attempted_retrievers=tuple(attempted),
            )
        raise RetrievalUnavailableError(
            f"no configured retriever could serve route: {plan.retrievers}"
        )

    async def _fallback(
        self,
        request: RetrievalRequest,
        selected: str,
        fallback: str,
    ) -> RetrievalResponse:
        """用显式 fallback 执行一次，并在响应保留原首选名称。"""
        try:
            retriever = self._retrievers[fallback]
        except KeyError as exc:
            raise ValueError(f"unknown fallback retriever: {fallback}") from exc
        response = await retriever.retrieve(request)
        return replace(
            response,
            retriever=fallback,
            fallback_from=selected,
            attempted_retrievers=(selected, fallback),
        )
