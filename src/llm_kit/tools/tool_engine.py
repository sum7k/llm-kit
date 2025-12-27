import inspect
import logging
from time import monotonic
from typing import Any

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .tool import ToolCall
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolEngine:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ) -> None:
        self.tool_registry = tool_registry
        self.metrics_hook = metrics_hook

    async def call_tool(self, tool_call: ToolCall) -> Any:
        logger.debug("Calling tool: %s", tool_call.tool_name)
        start = monotonic()
        tool = self.tool_registry.get(tool_call.tool_name)
        validated_args = tool.input_schema(**tool_call.arguments)

        # Check if handler is async
        if inspect.iscoroutinefunction(tool.handler):
            result = await tool.handler(validated_args)
        else:
            result = tool.handler(validated_args)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.TOOL_CALL_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.TOOL_CALLS_TOTAL, labels={"tool": tool_call.tool_name}
        )
        return result
