from __future__ import annotations

from codecraft.llm.base import LLMConfigError, LLMProvider, LLMProviderError


class LLMProviderRegistry:
    """按 provider name 管理可用的 LLMProvider。"""

    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        """创建注册表，并按输入顺序注册初始 Provider。

        Args:
            providers: 可选的 Provider 实例列表；名称必须非空且互不重复。

        Example:
            >>> from codecraft.llm.providers.mock import MockProvider
            >>> registry = LLMProviderRegistry([MockProvider()])
            >>> registry.get("MOCK").name
            'mock'
        """
        self._providers: dict[str, LLMProvider] = {}
        for provider in providers or ():
            self.register(provider)

    def register(self, provider: LLMProvider) -> None:
        """按规范化名称注册一个 Provider。

        Args:
            provider: 实现 ``LLMProvider`` 接口且具有 ``name`` 的实例。

        Returns:
            ``None``。

        Raises:
            ValueError: Provider 名称为空，或同名 Provider 已经注册。

        Example:
            >>> from codecraft.llm.providers.mock import MockProvider
            >>> registry = LLMProviderRegistry()
            >>> registry.register(MockProvider())
            >>> registry.get("mock").name
            'mock'
        """
        name = self._normalize_name(provider.name)
        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._providers:
            raise ValueError(f"provider already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        """按不区分大小写、忽略首尾空白的名称取得 Provider。

        Args:
            name: Provider 名称，例如 ``"openai"`` 或 ``" QWEN "``。

        Returns:
            注册时保存的同一个 Provider 实例。

        Raises:
            LLMConfigError: 规范化后的名称没有注册。
        """
        normalized = self._normalize_name(name)
        try:
            return self._providers[normalized]
        except KeyError as exc:
            raise LLMConfigError(f"provider not registered: {normalized}") from exc

    async def close(self) -> None:
        """按注册顺序的逆序关闭全部 Provider。

        Returns:
            所有 Provider 成功关闭时返回 ``None``。

        Raises:
            LLMProviderError: 至少一个 Provider 关闭失败；其余 Provider 仍会关闭。
        """
        errors: list[Exception] = []
        for provider in reversed(tuple(self._providers.values())):
            try:
                await provider.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise LLMProviderError(
                f"failed to close {len(errors)} model provider(s)"
            ) from errors[0]

    @staticmethod
    def _normalize_name(name: str) -> str:
        """把 Provider 名称规范为注册表键。

        Args:
            name: 原始名称。

        Returns:
            去除首尾空白并转换为小写的字符串。

        Example:
            >>> LLMProviderRegistry._normalize_name(" QWEN ")
            'qwen'
        """
        return name.strip().lower()
