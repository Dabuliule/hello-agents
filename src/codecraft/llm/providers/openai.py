from __future__ import annotations

from typing import Any

from codecraft.config.provider_defaults import OPENAI_API_KEY_ENV
from codecraft.llm.providers.responses import ResponsesProvider


class OpenAIProvider(ResponsesProvider):
    """使用原生 Responses API 的 OpenAI Provider。"""

    name = "openai"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = OPENAI_API_KEY_ENV,
        base_url: str | None = None,
    ) -> None:
        """配置 OpenAI Responses API Provider。

        Args:
            client: 可选的外部异步客户端，常用于测试或共享连接池。
            api_key: 显式 API Key。
            api_key_env: API Key 环境变量名，默认 ``OPENAI_API_KEY``。
            base_url: 可选的自定义 OpenAI 兼容地址。

        Example:
            >>> provider = OpenAIProvider(api_key_env="MY_OPENAI_KEY")
            >>> provider.api_key_env
            'MY_OPENAI_KEY'
        """
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="OPENAI_BASE_URL",
        )
