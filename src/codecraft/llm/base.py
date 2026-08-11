from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from codecraft.core.errors import ModelProviderError
from codecraft.llm.events import ModelEvent
from codecraft.llm.messages import ModelMessage
from codecraft.schema.tool import ToolSpec


class LLMConfigError(ModelProviderError):
    """LLM provider 缺少本地配置时抛出。"""

    def __init__(self, message: str) -> None:
        """创建错误码固定为 ``model_config_error`` 的配置异常。

        Args:
            message: 面向调用方的配置失败说明。

        Example:
            >>> LLMConfigError("API key is required").code
            'model_config_error'
        """
        super().__init__(message, code="model_config_error")


class LLMProviderError(ModelProviderError):
    """LLM provider 调用失败时抛出。"""

    def __init__(self, message: str) -> None:
        """创建错误码固定为 ``model_error`` 的供应商异常。

        Args:
            message: SDK、网络或供应商拒绝的可读说明。

        Example:
            >>> LLMProviderError("upstream unavailable").code
            'model_error'
        """
        super().__init__(message, code="model_error")


class LLMProtocolError(ModelProviderError):
    """LLM provider 违反内部事件协议时抛出。"""

    def __init__(self, message: str) -> None:
        """创建错误码固定为 ``model_protocol_error`` 的协议异常。

        Args:
            message: 响应结构或终态不符合内部协议的说明。

        Example:
            >>> LLMProtocolError("missing completed").code
            'model_protocol_error'
        """
        super().__init__(message, code="model_protocol_error")


class ModelRequest(BaseModel):
    """一次模型调用所需的最小、不可变输入。

    Example:
        >>> from codecraft.llm.messages import ModelRole, ModelTextMessage
        >>> request = ModelRequest(
        ...     model="test-model",
        ...     messages=(
        ...         ModelTextMessage(role=ModelRole.USER, content="你好"),
        ...     ),
        ... )
        >>> (request.model, request.max_output_tokens)
        ('test-model', 8192)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolSpec, ...] = ()
    max_output_tokens: int = Field(default=8192, ge=1)


class LLMProvider(ABC):
    """模型供应商适配器的统一生命周期与流式接口。"""

    name: str

    @abstractmethod
    def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelEvent]:
        """为一个模型请求创建异步成功事件流。

        Args:
            request: 供应商无关且不可变的模型请求。

        Returns:
            可由 ``async for`` 消费的 ``ModelEvent`` 异步迭代器。失败通过
            ``ModelProviderError`` 子类抛出，不编码成成功事件。
        """
        ...

    async def close(self) -> None:
        """释放 Provider 自己创建的资源；外部注入资源仍由调用方管理。"""
