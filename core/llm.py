"""HelloAgents统一LLM接口 - 基于OpenAI原生API"""
import os
from typing import Iterator, Literal, Optional, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .exceptions import HelloAgentsException

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai", "deepseek", "qwen", "modelscope",
    "kimi", "zhipu", "ollama", "vllm", "local", "auto"
]

# Provider 配置字典（集中管理，易于扩展）
PROVIDER_CONFIG = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-3.5-turbo",
        "env_check": "OPENAI_API_KEY",
        "url_keywords": ("api.openai.com",),
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "env_check": "DEEPSEEK_API_KEY",
        "url_keywords": ("api.deepseek.com",),
    },
    "qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen3-max-2026-01-23",
        "env_check": "DASHSCOPE_API_KEY",
        "url_keywords": ("dashscope.aliyuncs.com",),
    },
    "modelscope": {
        "env_key": "MODELSCOPE_API_KEY",
        "base_url": "https://api-inference.modelscope.cn/v1/",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "env_check": "MODELSCOPE_API_KEY",
        "url_keywords": ("modelscope.cn",),
    },
    "kimi": {
        "env_key": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "env_check": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "url_keywords": ("moonshot.cn",),
    },
    "zhipu": {
        "env_key": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4",
        "env_check": ("ZHIPU_API_KEY", "GLM_API_KEY"),
        "url_keywords": ("bigmodel.cn",),
    },
    "ollama": {
        "env_key": "OLLAMA_API_KEY",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "env_check": ("OLLAMA_API_KEY", "OLLAMA_HOST"),
        "default_api_key": "ollama",
        "url_keywords": (":11434", "ollama"),
    },
    "vllm": {
        "env_key": "VLLM_API_KEY",
        "base_url": "http://localhost:8000/v1",
        "default_model": "meta-llama/Llama-2-7b-chat-hf",
        "env_check": ("VLLM_API_KEY", "VLLM_HOST"),
        "default_api_key": "vllm",
        "url_keywords": (":8000", "vllm"),
    },
    "local": {
        "env_key": "LLM_API_KEY",
        "base_url": "http://localhost:8000/v1",
        "default_model": "local-model",
        "default_api_key": "local",
        "url_keywords": ("localhost", "127.0.0.1"),
    },
}


class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """

    def __init__(
            self,
            model: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            provider: Optional[SUPPORTED_PROVIDERS] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            timeout: Optional[int] = None,
            **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒【】
        """
        # 优先使用传入参数，如果未提供，则从环境变量加载
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # 自动检测provider或使用指定的provider
        self.provider = provider or self._auto_detect_provider(api_key, base_url)

        # 根据provider确定API密钥和base_url
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 验证必要参数
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

    @staticmethod
    def _normalize_messages(messages: list[dict[str, str]]) -> list[ChatCompletionMessageParam]:
        """适配OpenAI SDK的消息参数类型，保持现有调用方入参不变。"""
        return cast(list[ChatCompletionMessageParam], messages)

    @staticmethod
    def _auto_detect_provider(api_key: Optional[str], base_url: Optional[str]) -> str:
        """自动检测 LLM 提供商（优先级：特定环境变量 > API密钥 > base_url > 默认）"""
        # 1. 检查特定提供商的环境变量
        for provider, cfg in PROVIDER_CONFIG.items():
            if provider == "auto":
                continue
            env_checks = cfg.get("env_check")
            if isinstance(env_checks, str):
                env_checks = (env_checks,)
            if any(os.getenv(key) for key in env_checks):
                return provider

        # 2. 根据 API 密钥格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY", "").lower()
        if actual_api_key in {"ollama", "vllm", "local"}:
            return actual_api_key

        # 3. 根据 base_url 判断（采用最长匹配优先，避免子字符串冲突）
        actual_base_url = (base_url or os.getenv("LLM_BASE_URL", "")).lower()
        if actual_base_url:
            # 先收集所有匹配项
            matches = []
            for provider, cfg in PROVIDER_CONFIG.items():
                keywords = cfg.get("url_keywords", ())
                for keyword in keywords:
                    if keyword.lower() in actual_base_url:
                        # 记录 (provider, 关键词长度) 用于排序
                        matches.append((provider, len(keyword)))
                        break  # 一个 provider 只记录一条最优匹配

            # 返回关键词最长的匹配（最具体的）
            if matches:
                return max(matches, key=lambda x: x[1])[0]

        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """根据 provider 配置解析 API 密钥和 base_url"""
        # 使用配置字典，避免重复 if-elif
        provider_key = cast(str, self.provider or "auto")
        cfg = PROVIDER_CONFIG.get(provider_key, {})

        # 解析 API 密钥
        resolved_api_key = (
                api_key
                or self._get_env_keys(cfg.get("env_check"))
                or os.getenv("LLM_API_KEY")
                or cfg.get("default_api_key")
        )

        # 解析 base_url
        resolved_base_url = (
                base_url
                or os.getenv("LLM_BASE_URL")
                or cfg.get("base_url")
        )

        return resolved_api_key, resolved_base_url

    @staticmethod
    def _get_env_keys(env_checks: Optional[str | tuple[str, ...]]) -> Optional[str]:
        """从多个可能的环境变量名中获取第一个存在的值"""
        if not env_checks:
            return None
        env_list = (env_checks,) if isinstance(env_checks, str) else env_checks
        for key in env_list:
            value = os.getenv(key)
            if value:
                return value
        return None

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _get_default_model(self) -> str:
        """获取默认模型"""
        provider = cast(str, self.provider or "auto")
        # 优先从配置字典获取
        if provider in PROVIDER_CONFIG:
            return PROVIDER_CONFIG[provider].get("default_model", "gpt-3.5-turbo")

        # auto 模式：根据 base_url 推断
        base_url = os.getenv("LLM_BASE_URL", "").lower()
        for prov, cfg in PROVIDER_CONFIG.items():
            if cfg.get("base_url", "").lower() in base_url:
                return cfg.get("default_model", "gpt-3.5-turbo")

        # 默认返回 OpenAI 模型
        return "gpt-3.5-turbo"

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._normalize_messages(messages),
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._normalize_messages(messages),
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
