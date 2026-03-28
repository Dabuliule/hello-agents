"""工具基类。"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from core.exceptions import ToolException


class EmptyParams(BaseModel):
    """默认空参数模型。"""

    model_config = ConfigDict(extra="forbid")


class Tool(ABC):
    """工具基类，统一工具元数据、参数校验和执行生命周期。"""

    params_model: type[BaseModel] = EmptyParams

    def __init__(self, name: str, description: str):
        normalized_name = name.strip() if isinstance(name, str) else ""
        if not normalized_name:
            raise ToolException("Tool name 不能为空")

        normalized_description = description.strip() if isinstance(description, str) else ""
        if not normalized_description:
            raise ToolException("Tool description 不能为空")

        self.name = normalized_name
        self.description = normalized_description

    def validate_params(self, params: dict[str, Any] | BaseModel | None) -> dict[str, Any]:
        """使用 Pydantic 模型校验并规范化参数。"""
        raw_params: dict[str, Any]
        if params is None:
            raw_params = {}
        elif isinstance(params, BaseModel):
            raw_params = params.model_dump()
        elif isinstance(params, dict):
            raw_params = params
        else:
            raise ToolException("Tool params 必须是 dict 或 pydantic BaseModel")

        try:
            validated = self.params_model.model_validate(raw_params)
        except ValidationError as exc:
            raise ToolException(f"工具 {self.name} 参数校验失败: {exc}") from exc
        return validated.model_dump()

    def execute(self, params: dict[str, Any] | BaseModel | None = None) -> Any:
        """工具统一执行入口。"""
        normalized_params = self.validate_params(params)
        try:
            return self._run(normalized_params)
        except ToolException:
            raise
        except Exception as exc:
            raise ToolException(f"工具 {self.name} 执行失败: {exc}") from exc

    @abstractmethod
    def _run(self, params: dict[str, Any]) -> Any:
        """工具子类具体执行逻辑。"""

    def to_dict(self) -> dict[str, Any]:
        """导出工具描述，便于注册到函数调用协议。"""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.params_model.model_json_schema(),
        }
