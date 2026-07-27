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
        super().__init__(message, code="model_config_error")


class LLMProviderError(ModelProviderError):
    """LLM provider 调用失败时抛出。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="model_error")


class LLMProtocolError(ModelProviderError):
    """LLM provider 违反内部事件协议时抛出。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="model_protocol_error")


class ModelRequest(BaseModel):
    """一次模型调用所需的最小、不可变输入。"""

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
    ) -> AsyncIterator[ModelEvent]: ...

    async def close(self) -> None:
        """释放 Provider 自己创建的资源；外部注入资源仍由调用方管理。"""
