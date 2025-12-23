import logging

from .tool import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            logger.error("Tool not found: %s", name)
            raise KeyError(f"Tool '{name}' not found")

    def remove(self, name: str) -> None:
        try:
            del self._tools[name]
            logger.debug("Removed tool: %s", name)
        except KeyError:
            logger.error("Cannot remove tool, not found: %s", name)
            raise KeyError(f"Tool '{name}' not found")

    def list(self) -> dict[str, Tool]:
        # return a shallow copy to avoid mutation
        return dict(self._tools)
