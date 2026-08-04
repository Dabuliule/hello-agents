from __future__ import annotations

from typing import Any

from codecraft.llm.providers.chat import ChatCompletionsProvider


class DeepSeekProvider(ChatCompletionsProvider):
    """通过 OpenAI 兼容端点调用 DeepSeek Chat Completions。"""

    name = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = "DEEPSEEK_API_KEY",
        base_url: str | None = None,
    ) -> None:
        """配置 DeepSeek Chat Completions 兼容端点。"""
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url=self.DEFAULT_BASE_URL,
        )
