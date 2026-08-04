from __future__ import annotations

from typing import Any

from codecraft.llm.providers.responses import ResponsesProvider


class OpenAIProvider(ResponsesProvider):
    """使用原生 Responses API 的 OpenAI Provider。"""

    name = "openai"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        """配置 OpenAI Responses API 客户端或接收外部注入的兼容客户端。"""
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="OPENAI_BASE_URL",
        )
