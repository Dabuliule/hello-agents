"""内置计算器工具。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from core.exceptions import ToolException
from tools.base import Tool


class CalculatorParams(BaseModel):
    """计算器参数。"""

    model_config = ConfigDict(extra="forbid")

    left: float
    right: float
    operator: Literal["+", "-", "*", "/"]


class CalculatorTool(Tool):
    """执行基础四则运算。"""

    params_model = CalculatorParams

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行基础四则运算（+、-、*、/）",
        )

    def _run(self, params: dict[str, float | str]) -> float:
        left = float(params["left"])
        right = float(params["right"])
        operator = str(params["operator"])

        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            if right == 0:
                raise ToolException("除数不能为 0")
            return left / right

        raise ToolException(f"不支持的运算符: {operator}")
