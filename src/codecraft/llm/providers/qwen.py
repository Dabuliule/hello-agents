from __future__ import annotations

from typing import Any

from codecraft.llm.providers.chat import ChatCompletionsProvider


class QwenProvider(ChatCompletionsProvider):
    """通过 DashScope 兼容端点调用 Qwen Chat Completions。"""

    name = "qwen"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = "DASHSCOPE_API_KEY",
        base_url: str | None = None,
    ) -> None:
        """配置 Qwen 的 DashScope Chat Completions 兼容端点。"""
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="QWEN_BASE_URL",
            default_base_url=self.DEFAULT_BASE_URL,
        )
