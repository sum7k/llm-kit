# src/llm_kit/llms/openai.py

import json
import logging
from time import monotonic
from typing import Any, Literal

from openai import NOT_GIVEN, AsyncOpenAI, OpenAIError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook
from llm_kit.tools.tool import Tool

from ._tool_schema import tools_to_openai_schema
from .base import LLMClient, LLMResponse, Message, ToolCall, Usage

logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """OpenAI LLM client.

    Stateless. Transport-only retries. No behavior.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        timeout: float = 30.0,
        max_retries: int = 3,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ):
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_retries = max_retries
        self.metrics_hook = metrics_hook
        logger.info(
            "Initialized OpenAILLMClient with model=%s, timeout=%s",
            model,
            timeout,
        )

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start = monotonic()

        # Convert to provider format (internal only - never leaks)
        openai_messages = self._convert_messages(messages)
        openai_tools = tools_to_openai_schema(tools) if tools else None

        logger.debug(
            "Calling OpenAI: model=%s, messages=%d, tools=%d",
            self._model,
            len(messages),
            len(tools) if tools else 0,
        )

        raw = await self._call_api(
            messages=openai_messages,
            tools=openai_tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        elapsed_ms = 1000 * (monotonic() - start)

        # Normalize immediately - provider objects never escape
        response = self._normalize_response(raw, elapsed_ms)

        # Metrics
        self.metrics_hook.record_latency(names.LLM_COMPLETION_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.LLM_REQUESTS_TOTAL,
            labels={"provider": "openai", "model": self._model},
        )
        self.metrics_hook.increment(
            names.LLM_TOKENS_PROMPT, response.usage.prompt_tokens
        )
        self.metrics_hook.increment(
            names.LLM_TOKENS_COMPLETION, response.usage.completion_tokens
        )
        self.metrics_hook.increment(names.LLM_TOKENS_TOTAL, response.usage.total_tokens)

        logger.info(
            "OpenAI completion: finish=%s, tokens=%d, latency=%.0fms",
            response.finish_reason,
            response.usage.total_tokens,
            elapsed_ms,
        )

        return response

    async def _call_api(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Call OpenAI API with transport-only retries."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception_type(OpenAIError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    tools=tools if tools else NOT_GIVEN,  # type: ignore[arg-type]
                    max_tokens=max_tokens if max_tokens else NOT_GIVEN,  # type: ignore[arg-type]
                )

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to OpenAI format.

        Internal only. Provider format never leaks outside.
        """
        result = []
        for m in messages:
            msg: dict = {"role": m.role.value, "content": m.content}
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            result.append(msg)
        return result

    def _normalize_response(self, raw: Any, latency_ms: float) -> LLMResponse:
        """Normalize OpenAI response to LLMResponse.

        This is the boundary. Raw provider objects stop here.
        """
        choice = raw.choices[0]

        # Parse tool calls if present
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool call arguments: %s",
                        tc.function.arguments,
                    )
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        # Map finish reason
        finish_reason: Literal["stop", "tool_calls", "length", "error"]
        if tool_calls:
            finish_reason = "tool_calls"
        elif choice.finish_reason == "stop":
            finish_reason = "stop"
        elif choice.finish_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = "error"

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=raw.usage.prompt_tokens,
                completion_tokens=raw.usage.completion_tokens,
                total_tokens=raw.usage.total_tokens,
            ),
            latency_ms=latency_ms,
        )
