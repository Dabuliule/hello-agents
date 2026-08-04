from __future__ import annotations

import inspect
import os
from typing import Any

from codecraft.llm.base import LLMConfigError, LLMProvider


class OpenAIClientProvider(LLMProvider):
    """管理 OpenAI Python client 的创建与所有权。

    Provider 只关闭自己延迟创建的 client。测试或上层注入的 client 仍由注入方
    管理，避免共享连接被某个 Provider 意外关闭。
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None,
        base_url: str | None = None,
        base_url_env: str | None = None,
        default_base_url: str | None = None,
    ) -> None:
        """保存客户端来源和连接配置；真正的 SDK 客户端按需延迟创建。"""
        self._client_instance = client
        self._owns_client = client is None
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._base_url_env = base_url_env
        self._default_base_url = default_base_url

    @property
    def api_key_env(self) -> str | None:
        """返回 Provider 将读取的 API Key 环境变量名称。"""
        return self._api_key_env

    @property
    def base_url(self) -> str | None:
        """返回显式配置的 API 地址；不展开环境变量或默认地址。"""
        return self._base_url

    def _client(self) -> Any:
        """返回进程内复用的 client，首次调用时才读取密钥并创建连接池。"""
        if self._client_instance is None:
            self._client_instance = self._create_client()
        return self._client_instance

    def _create_client(self) -> Any:
        """解析密钥和地址优先级，并创建一个异步 OpenAI 兼容客户端。"""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMConfigError("openai package is not installed") from exc

        api_key = self._api_key or (
            os.getenv(self._api_key_env) if self._api_key_env else None
        )
        if not api_key:
            source = self._api_key_env or "an explicit api_key"
            raise LLMConfigError(f"{source} is required for {type(self).__name__}")

        base_url = self._base_url
        if base_url is None and self._base_url_env:
            base_url = os.getenv(self._base_url_env)
        if base_url is None:
            base_url = self._default_base_url

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    async def close(self) -> None:
        """关闭 Provider 自己创建的客户端，保留外部注入客户端的所有权。"""
        if not self._owns_client or self._client_instance is None:
            return

        client = self._client_instance
        self._client_instance = None
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result
