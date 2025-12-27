# src/llm_kit/llms/config.py

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM clients.

    Immutable. Explicit. No magic defaults from environment.
    """

    provider: Literal["openai", "anthropic"]
    model: str
    api_key: str | None = None  # Falls back to provider's env var
    timeout: float = 30.0
    max_retries: int = 3
