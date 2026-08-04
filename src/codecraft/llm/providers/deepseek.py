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
        """配置 DeepSeek Chat Completions Provider。

        Args:
            client: 可选的外部异步客户端，常用于测试或共享连接池。
            api_key: 显式 API Key。
            api_key_env: API Key 环境变量名，默认 ``DEEPSEEK_API_KEY``。
            base_url: 可选的自定义兼容地址；未传时使用 DeepSeek 默认地址。

        Example:
            >>> provider = DeepSeekProvider(api_key_env="MY_DEEPSEEK_KEY")
            >>> provider.api_key_env
            'MY_DEEPSEEK_KEY'
        """
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url=self.DEFAULT_BASE_URL,
        )
