"""消息系统"""
from datetime import datetime
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（OpenAI API格式）"""
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


if __name__ == "__main__":
    # 测试Message类
    msg1 = Message(content="你好，世界！", role="user")
    msg2 = Message(content="你好！有什么我可以帮助你的吗？", role="assistant")

    print(msg1)
    print(msg2)

    print(msg1.model_dump())
    print(msg2.model_dump())

    print(msg1.to_dict())
    print(msg2.to_dict())
