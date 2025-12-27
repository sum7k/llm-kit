# src/llm_kit/llms/__init__.py

"""LLM client layer for llm-kit.

Provides a thin, stateless abstraction over LLM providers.

Design principles:
- Stateless: Every call receives full message list
- Transport only: Retries only on network/rate-limit errors
- No behavior: No loops, no prompt fixing, no "smart" retries
- No leakage: Provider objects never escape the adapter

Example:
    >>> from llm_kit.llms import create_llm_client, LLMConfig, Message, Role
    >>>
    >>> config = LLMConfig(provider="openai", model="gpt-4o")
    >>> client = create_llm_client(config)
    >>>
    >>> response = client.complete(
    ...     messages=[Message(role=Role.USER, content="Hello!")]
    ... )
    >>> print(response.content)
"""

from .base import LLMClient, LLMResponse, Message, Role, ToolCall, Usage
from .config import LLMConfig
from .factory import create_llm_client

__all__ = [
    # Factory
    "create_llm_client",
    # Protocol
    "LLMClient",
    # Config
    "LLMConfig",
    # Types
    "Message",
    "Role",
    "ToolCall",
    "LLMResponse",
    "Usage",
]
