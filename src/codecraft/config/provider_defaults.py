"""内置模型 Provider 的连接默认值。

本模块是 CLI 配置生成器、Session 配置解析和具体 Provider 适配器之间的共享
真值源。这里只保存环境变量的**名称**和公开 API endpoint，不读取、保存或输出
任何真实密钥。
"""

from __future__ import annotations


QWEN_API_KEY_ENV = "DASHSCOPE_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_API_KEY_ENV_BY_PROVIDER = {
    "qwen": QWEN_API_KEY_ENV,
    "openai": OPENAI_API_KEY_ENV,
    "deepseek": DEEPSEEK_API_KEY_ENV,
}
_BASE_URL_BY_PROVIDER = {
    "qwen": QWEN_BASE_URL,
    "deepseek": DEEPSEEK_BASE_URL,
}


def default_api_key_env(provider: str) -> str | None:
    """返回已知 Provider 的标准 API Key 环境变量名。

    未知 Provider 返回 ``None``，由它自己的配置边界给出错误。返回值只是例如
    ``DASHSCOPE_API_KEY`` 这样的变量名，调用方仍需在真正请求模型时读取变量值。
    """
    return _API_KEY_ENV_BY_PROVIDER.get(provider)


def default_base_url(provider: str) -> str | None:
    """返回需要显式兼容端点的内置 Provider 默认地址。

    OpenAI 返回 ``None``，交给官方 SDK 使用其原生默认地址；Qwen 和 DeepSeek
    返回各自的 OpenAI-compatible endpoint。
    """
    return _BASE_URL_BY_PROVIDER.get(provider)
