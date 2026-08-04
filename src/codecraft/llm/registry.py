from __future__ import annotations

from codecraft.llm.base import LLMConfigError, LLMProvider, LLMProviderError


class LLMProviderRegistry:
    """按 provider name 管理可用的 LLMProvider。"""

    def __init__(self, providers: list[LLMProvider] | None = None) -> None:
        """创建空注册表，并按输入顺序注册初始 Provider。"""
        self._providers: dict[str, LLMProvider] = {}
        for provider in providers or ():
            self.register(provider)

    def register(self, provider: LLMProvider) -> None:
        """注册一个 provider，并拒绝空名称或重复名称。"""
        name = self._normalize_name(provider.name)
        if not name:
            raise ValueError("provider name must not be empty")
        if name in self._providers:
            raise ValueError(f"provider already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        """按名称取 provider。"""
        normalized = self._normalize_name(name)
        try:
            return self._providers[normalized]
        except KeyError as exc:
            raise LLMConfigError(f"provider not registered: {normalized}") from exc

    async def close(self) -> None:
        """关闭全部 Provider，并保证单个关闭失败不会跳过其余实例。"""
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
        """把 Provider 名称规范为去除首尾空白的小写键。"""
        return name.strip().lower()
