# tests/unit/llms/test_anthropic.py

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_kit.llms.anthropic import AnthropicLLMClient
from llm_kit.llms.base import Message, Role
from llm_kit.tools.tool import Tool


class WeatherInput(BaseModel):
    city: str


@pytest.fixture
def mock_anthropic_response() -> MagicMock:
    """Create a mock Anthropic response."""
    response = MagicMock()

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello! How can I help you?"

    response.content = [text_block]
    response.stop_reason = "end_turn"
    response.usage.input_tokens = 10
    response.usage.output_tokens = 8
    return response


@pytest.fixture
def mock_anthropic_tool_response() -> MagicMock:
    """Create a mock Anthropic response with tool use."""
    response = MagicMock()

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "toolu_123"
    tool_block.name = "get_weather"
    tool_block.input = {"city": "Tokyo"}

    response.content = [tool_block]
    response.stop_reason = "tool_use"
    response.usage.input_tokens = 15
    response.usage.output_tokens = 12
    return response


class TestAnthropicLLMClient:
    @pytest.mark.asyncio
    async def test_complete_basic(self, mock_anthropic_response: MagicMock) -> None:
        """Test basic completion without tools."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            client = AnthropicLLMClient(api_key="test-key")
            response = await client.complete(
                messages=[Message(role=Role.USER, content="Hello!")]
            )

            assert response.content == "Hello! How can I help you?"
            assert response.finish_reason == "stop"
            assert response.tool_calls == []
            assert response.usage.total_tokens == 18
            assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_complete_with_tools(
        self, mock_anthropic_tool_response: MagicMock
    ) -> None:
        """Test completion with tool calls."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_anthropic_tool_response
            mock_anthropic.return_value = mock_client

            client = AnthropicLLMClient(api_key="test-key")

            tool = Tool(
                name="get_weather",
                description="Get weather for a city",
                input_schema=WeatherInput,
                handler=lambda x: f"Weather in {x.city}",
            )

            response = await client.complete(
                messages=[Message(role=Role.USER, content="What's the weather?")],
                tools=[tool],
            )

            assert response.content is None
            assert response.finish_reason == "tool_calls"
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].id == "toolu_123"
            assert response.tool_calls[0].name == "get_weather"
            assert response.tool_calls[0].arguments == {"city": "Tokyo"}

    def test_system_message_extraction(self) -> None:
        """Test that system messages are extracted correctly."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic"):
            client = AnthropicLLMClient(api_key="test-key")

            messages = [
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="Hello"),
            ]

            system, non_system = client._extract_system(messages)

            assert system == "You are helpful."
            assert len(non_system) == 1
            assert non_system[0].role == Role.USER

    def test_tool_result_conversion(self) -> None:
        """Test that tool results are converted to Anthropic format."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic"):
            client = AnthropicLLMClient(api_key="test-key")

            messages = [
                Message(role=Role.TOOL, content="22°C sunny", tool_call_id="toolu_123"),
            ]

            converted = client._convert_messages(messages)

            assert converted[0]["role"] == "user"
            assert converted[0]["content"][0]["type"] == "tool_result"
            assert converted[0]["content"][0]["tool_use_id"] == "toolu_123"
            assert converted[0]["content"][0]["content"] == "22°C sunny"

    @pytest.mark.asyncio
    async def test_metrics_hook_called(
        self, mock_anthropic_response: MagicMock
    ) -> None:
        """Test that metrics hook is called."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_anthropic_response
            mock_anthropic.return_value = mock_client

            metrics_hook = MagicMock()
            client = AnthropicLLMClient(api_key="test-key", metrics_hook=metrics_hook)

            await client.complete(messages=[Message(role=Role.USER, content="Hi")])

            metrics_hook.record_latency.assert_called_once()
            call_args = metrics_hook.record_latency.call_args
            assert call_args[0][0] == "llm_completion_duration"

    @pytest.mark.asyncio
    async def test_max_tokens_finish_reason(self) -> None:
        """Test that max_tokens stop reason is mapped correctly."""
        with patch("llm_kit.llms.anthropic.AsyncAnthropic") as mock_anthropic:
            response = MagicMock()
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Truncated..."
            response.content = [text_block]
            response.stop_reason = "max_tokens"
            response.usage.input_tokens = 10
            response.usage.output_tokens = 100

            mock_client = AsyncMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.return_value = mock_client

            client = AnthropicLLMClient(api_key="test-key")
            result = await client.complete(
                messages=[Message(role=Role.USER, content="test")]
            )

            assert result.finish_reason == "length"
