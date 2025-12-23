from .tool import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' not found")

    def remove(self, name: str) -> None:
        try:
            del self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' not found")

    def list(self) -> dict[str, Tool]:
        # return a shallow copy to avoid mutation
        return dict(self._tools)
