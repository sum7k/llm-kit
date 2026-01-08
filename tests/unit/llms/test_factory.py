# tests/unit/llms/test_factory.py

from unittest.mock import patch

import pytest

from llm_kit.llms import LLMConfig, create_llm_client
from llm_kit.llms.anthropic import AnthropicLLMClient
from llm_kit.llms.openai import OpenAILLMClient


class TestFactory:
    def test_create_openai_client(self) -> None:
        """Test creating OpenAI client."""
        with patch("llm_kit.llms.openai.AsyncOpenAI"):
            config = LLMConfig(provider="openai", model="gpt-4o", api_key="test")
            client = create_llm_client(config)
            assert isinstance(client, OpenAILLMClient)

    def test_create_anthropic_client(self) -> None:
        """Test creating Anthropic client."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic"):
            config = LLMConfig(
                provider="anthropic", model="claude-sonnet-4-20250514", api_key="test"
            )
            client = create_llm_client(config)
            assert isinstance(client, AnthropicLLMClient)

    def test_unknown_provider_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        config = LLMConfig(provider="unknown", model="model")  # type: ignore
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_llm_client(config)

    def test_config_values_passed_through(self) -> None:
        """Test that config values are passed to client."""
        with patch("llm_kit.llms.openai.AsyncOpenAI") as mock_openai:
            config = LLMConfig(
                provider="openai",
                model="gpt-4-turbo",
                api_key="my-key",
                timeout=60.0,
                max_retries=5,
            )
            client = create_llm_client(config)

            assert client._model == "gpt-4-turbo"
            assert client._max_retries == 5
            mock_openai.assert_called_once_with(api_key="my-key", timeout=60.0)
