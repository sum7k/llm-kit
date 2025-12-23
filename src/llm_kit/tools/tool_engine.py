import logging
from typing import Any

from .tool import ToolCall
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolEngine:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def call_tool(self, tool_call: ToolCall) -> Any:
        logger.debug("Calling tool: %s", tool_call.tool_name)
        tool = self.tool_registry.get(tool_call.tool_name)
        validated_args = tool.input_schema(**tool_call.arguments)
        return tool.handler(validated_args)
