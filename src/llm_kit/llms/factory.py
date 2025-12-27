# src/llm_kit/llms/factory.py

from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import LLMClient
from .config import LLMConfig


def create_llm_client(
    config: LLMConfig,
    metrics_hook: MetricsHook = NoOpMetricsHook(),
) -> LLMClient:
    """Create an LLM client from config.

    Args:
        config: LLM configuration specifying provider, model, etc.
        metrics_hook: Optional metrics hook for observability.

    Returns:
        Configured LLMClient implementation.

    Raises:
        ValueError: If provider is unknown.

    Example:
        >>> config = LLMConfig(provider="openai", model="gpt-4o")
        >>> client = create_llm_client(config)
        >>> response = client.complete(messages=[...])
    """
    if config.provider == "openai":
        from .openai import OpenAILLMClient

        return OpenAILLMClient(
            api_key=config.api_key,
            model=config.model,
            timeout=config.timeout,
            max_retries=config.max_retries,
            metrics_hook=metrics_hook,
        )

    if config.provider == "anthropic":
        from .anthropic import AnthropicLLMClient

        return AnthropicLLMClient(
            api_key=config.api_key,
            model=config.model,
            timeout=config.timeout,
            max_retries=config.max_retries,
            metrics_hook=metrics_hook,
        )

    raise ValueError(f"Unknown LLM provider: {config.provider}")
