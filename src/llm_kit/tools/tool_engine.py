from typing import Any

from .tool import ToolCall
from .tool_registry import ToolRegistry


class ToolEngine:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def call_tool(self, tool_call: ToolCall) -> Any:
        tool = self.tool_registry.get(tool_call.tool_name)
        validated_args = tool.input_schema(**tool_call.arguments)
        return tool.handler(validated_args)
