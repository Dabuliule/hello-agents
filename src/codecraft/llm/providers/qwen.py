from __future__ import annotations

from typing import Any

from codecraft.config.provider_defaults import QWEN_API_KEY_ENV, QWEN_BASE_URL
from codecraft.llm.providers.chat import ChatCompletionsProvider


class QwenProvider(ChatCompletionsProvider):
    """通过 DashScope 兼容端点调用 Qwen Chat Completions。"""

    name = "qwen"
    DEFAULT_BASE_URL = QWEN_BASE_URL

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = QWEN_API_KEY_ENV,
        base_url: str | None = None,
    ) -> None:
        """配置 Qwen 的 DashScope Chat Completions Provider。

        Args:
            client: 可选的外部异步客户端，常用于测试或共享连接池。
            api_key: 显式 API Key。
            api_key_env: API Key 环境变量名，默认 ``DASHSCOPE_API_KEY``。
            base_url: 可选的自定义兼容地址；未传时使用 DashScope 默认地址。

        Example:
            >>> provider = QwenProvider(api_key_env="MY_QWEN_KEY")
            >>> provider.api_key_env
            'MY_QWEN_KEY'
        """
        super().__init__(
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            base_url_env="QWEN_BASE_URL",
            default_base_url=self.DEFAULT_BASE_URL,
        )
