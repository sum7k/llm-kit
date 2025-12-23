from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class Tool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        handler: Callable,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


class ToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any]

    class Config:
        extra = "forbid"
