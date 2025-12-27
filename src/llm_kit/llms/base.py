# src/llm_kit/llms/base.py

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

from llm_kit.observability.base import MetricsHook
from llm_kit.tools.tool import Tool


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single message in the conversation.

    Immutable. Stateless. Provider-agnostic.
    """

    role: Role
    content: str
    tool_call_id: str | None = None  # Required when role=TOOL


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call from LLM response.

    Provider-agnostic representation. Never exposes raw provider objects.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    """Token usage for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response.

    Provider details never leak outside the adapter.
    This is the only type callers ever see.
    """

    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "error"]
    usage: Usage
    latency_ms: float


class LLMClient(Protocol):
    """Protocol for LLM clients.

    Design principles:
    - Stateless: Every call receives full message list
    - Transport only: Retries only on network/rate-limit errors
    - No behavior: No loops, no prompt fixing, no "smart" retries
    - No leakage: Provider objects never escape the adapter
    """

    metrics_hook: MetricsHook

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Single completion. Stateless. Full message list required.

        Args:
            messages: Complete conversation history. No internal state.
            tools: Optional list of tools the model can call.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in response.

        Returns:
            Normalized LLMResponse. Provider details never leak.

        Raises:
            Provider-specific errors after retry exhaustion.

        Note:
            Retries only on transport errors (network, rate-limit).
            Never retries on "bad" model output - that's the caller's problem.
        """
        ...
