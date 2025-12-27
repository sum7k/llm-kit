# src/llm_kit/llms/anthropic.py

import logging
from time import monotonic
from typing import Any, Literal

from anthropic import NOT_GIVEN, Anthropic, APIError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_kit.observability.base import MetricsHook, NoOpMetricsHook
from llm_kit.tools.tool import Tool

from ._tool_schema import tools_to_anthropic_schema
from .base import LLMClient, LLMResponse, Message, Role, ToolCall, Usage

logger = logging.getLogger(__name__)


class AnthropicLLMClient(LLMClient):
    """Anthropic LLM client.

    Stateless. Transport-only retries. No behavior.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 30.0,
        max_retries: int = 3,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ):
        self._client = Anthropic(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries
        self.metrics_hook = metrics_hook
        logger.info(
            "Initialized AnthropicLLMClient with model=%s, timeout=%s",
            model,
            timeout,
        )

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start = monotonic()

        # Extract system message (Anthropic handles it separately)
        system_content, non_system_messages = self._extract_system(messages)

        # Convert to provider format (internal only - never leaks)
        anthropic_messages = self._convert_messages(non_system_messages)
        anthropic_tools = tools_to_anthropic_schema(tools) if tools else None

        logger.debug(
            "Calling Anthropic: model=%s, messages=%d, tools=%d",
            self._model,
            len(messages),
            len(tools) if tools else 0,
        )

        raw = self._call_api(
            system=system_content,
            messages=anthropic_messages,
            tools=anthropic_tools,
            temperature=temperature,
            max_tokens=max_tokens or 4096,  # Anthropic requires max_tokens
        )

        elapsed_ms = 1000 * (monotonic() - start)

        # Normalize immediately - provider objects never escape
        response = self._normalize_response(raw, elapsed_ms)

        # Metrics
        self.metrics_hook.record_latency("llm_completion_duration", elapsed_ms)

        logger.info(
            "Anthropic completion: finish=%s, tokens=%d, latency=%.0fms",
            response.finish_reason,
            response.usage.total_tokens,
            elapsed_ms,
        )

        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
        retry=retry_if_exception_type(APIError),  # Transport only
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_api(
        self,
        *,
        system: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Call Anthropic API with transport-only retries."""
        return self._client.messages.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            system=system if system else NOT_GIVEN,
            tools=tools if tools else NOT_GIVEN,  # type: ignore[arg-type]
        )

    def _extract_system(
        self, messages: list[Message]
    ) -> tuple[str | None, list[Message]]:
        """Extract system message from message list.

        Anthropic requires system message as a separate parameter.
        """
        system_content = None
        non_system = []

        for m in messages:
            if m.role == Role.SYSTEM:
                system_content = m.content
            else:
                non_system.append(m)

        return system_content, non_system

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to Anthropic format.

        Internal only. Provider format never leaks outside.
        """
        result = []
        for m in messages:
            if m.role == Role.TOOL:
                # Anthropic tool results have a different structure
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            else:
                result.append({"role": m.role.value, "content": m.content})
        return result

    def _normalize_response(self, raw: Any, latency_ms: float) -> LLMResponse:
        """Normalize Anthropic response to LLMResponse.

        This is the boundary. Raw provider objects stop here.
        """
        # Extract content and tool calls from content blocks
        text_content: str | None = None
        tool_calls: list[ToolCall] = []

        for block in raw.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        # Map finish reason
        finish_reason: Literal["stop", "tool_calls", "length", "error"]
        if tool_calls:
            finish_reason = "tool_calls"
        elif raw.stop_reason == "end_turn":
            finish_reason = "stop"
        elif raw.stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "error"

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
            ),
            latency_ms=latency_ms,
        )
