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
        """保存客户端来源和连接配置，延迟创建真正的 SDK 客户端。

        Args:
            client: 外部管理的 OpenAI 兼容异步客户端；传入后 Provider 不关闭它。
            api_key: 显式 API Key，优先级高于环境变量。
            api_key_env: API Key 环境变量名；可以为 ``None``。
            base_url: 显式 API 地址，优先级高于环境变量和默认地址。
            base_url_env: API 地址环境变量名。
            default_base_url: 前两种地址都不存在时使用的 Provider 默认地址。
        """
        self._client_instance = client
        self._owns_client = client is None
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._base_url = base_url
        self._base_url_env = base_url_env
        self._default_base_url = default_base_url

    @property
    def api_key_env(self) -> str | None:
        """返回 Provider 将读取的 API Key 环境变量名称，未配置时返回 ``None``。"""
        return self._api_key_env

    @property
    def base_url(self) -> str | None:
        """返回显式 API 地址，未配置时返回 ``None``；不展开后备地址。"""
        return self._base_url

    def _client(self) -> Any:
        """取得当前 Provider 复用的异步 SDK 客户端。

        Returns:
            构造时注入的客户端，或首次调用时创建并缓存的 ``AsyncOpenAI``。

        Raises:
            LLMConfigError: SDK 未安装，或者无法解析 API Key。
        """
        if self._client_instance is None:
            self._client_instance = self._create_client()
        return self._client_instance

    def _create_client(self) -> Any:
        """解析密钥和地址优先级，并创建异步 OpenAI 兼容客户端。

        Returns:
            使用解析后 ``api_key`` 和可选 ``base_url`` 创建的 ``AsyncOpenAI``。

        Raises:
            LLMConfigError: ``openai`` 包未安装，或显式值和环境变量都没有密钥。
        """
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
        """关闭 Provider 自己创建的客户端。

        Returns:
            ``None``。外部注入的客户端不会被关闭；同步和异步 ``close`` 均支持。
        """
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
