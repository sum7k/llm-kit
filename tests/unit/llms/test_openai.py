# tests/unit/llms/test_openai.py

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_kit.llms.base import Message, Role
from llm_kit.llms.openai import OpenAILLMClient
from llm_kit.tools.tool import Tool


class WeatherInput(BaseModel):
    city: str


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Create a mock OpenAI response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello! How can I help you?"
    response.choices[0].message.tool_calls = None
    response.choices[0].finish_reason = "stop"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 8
    response.usage.total_tokens = 18
    return response


@pytest.fixture
def mock_openai_tool_response() -> MagicMock:
    """Create a mock OpenAI response with tool calls."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = None
    response.choices[0].finish_reason = "tool_calls"

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = '{"city": "Tokyo"}'

    response.choices[0].message.tool_calls = [tool_call]
    response.usage.prompt_tokens = 15
    response.usage.completion_tokens = 12
    response.usage.total_tokens = 27
    return response


class TestOpenAILLMClient:
    def test_complete_basic(self, mock_openai_response: MagicMock) -> None:
        """Test basic completion without tools."""
        with patch("llm_kit.llms.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            client = OpenAILLMClient(api_key="test-key", model="gpt-4o")
            response = client.complete(
                messages=[Message(role=Role.USER, content="Hello!")]
            )

            assert response.content == "Hello! How can I help you?"
            assert response.finish_reason == "stop"
            assert response.tool_calls == []
            assert response.usage.total_tokens == 18
            assert response.latency_ms > 0

    def test_complete_with_tools(self, mock_openai_tool_response: MagicMock) -> None:
        """Test completion with tool calls."""
        with patch("llm_kit.llms.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_tool_response
            mock_openai.return_value = mock_client

            client = OpenAILLMClient(api_key="test-key")

            tool = Tool(
                name="get_weather",
                description="Get weather for a city",
                input_schema=WeatherInput,
                handler=lambda x: f"Weather in {x.city}",
            )

            response = client.complete(
                messages=[Message(role=Role.USER, content="What's the weather?")],
                tools=[tool],
            )

            assert response.content is None
            assert response.finish_reason == "tool_calls"
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].id == "call_123"
            assert response.tool_calls[0].name == "get_weather"
            assert response.tool_calls[0].arguments == {"city": "Tokyo"}

    def test_message_conversion(self) -> None:
        """Test message conversion to OpenAI format."""
        with patch("llm_kit.llms.openai.OpenAI"):
            client = OpenAILLMClient(api_key="test-key")

            messages = [
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
                Message(role=Role.TOOL, content="Result", tool_call_id="call_123"),
            ]

            converted = client._convert_messages(messages)

            assert converted[0] == {"role": "system", "content": "You are helpful."}
            assert converted[1] == {"role": "user", "content": "Hello"}
            assert converted[2] == {"role": "assistant", "content": "Hi there!"}
            assert converted[3] == {
                "role": "tool",
                "content": "Result",
                "tool_call_id": "call_123",
            }

    def test_metrics_hook_called(self, mock_openai_response: MagicMock) -> None:
        """Test that metrics hook is called."""
        with patch("llm_kit.llms.openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_openai_response
            mock_openai.return_value = mock_client

            metrics_hook = MagicMock()
            client = OpenAILLMClient(api_key="test-key", metrics_hook=metrics_hook)

            client.complete(messages=[Message(role=Role.USER, content="Hi")])

            metrics_hook.record_latency.assert_called_once()
            call_args = metrics_hook.record_latency.call_args
            assert call_args[0][0] == "llm_completion_duration"
            assert call_args[0][1] > 0

    def test_malformed_tool_arguments_handled(self) -> None:
        """Test that malformed JSON in tool arguments is handled gracefully."""
        with patch("llm_kit.llms.openai.OpenAI") as mock_openai:
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = None
            response.choices[0].finish_reason = "tool_calls"

            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "get_weather"
            tool_call.function.arguments = "invalid json"

            response.choices[0].message.tool_calls = [tool_call]
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 5
            response.usage.total_tokens = 15

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.return_value = mock_client

            client = OpenAILLMClient(api_key="test-key")
            result = client.complete(messages=[Message(role=Role.USER, content="test")])

            # Should not raise, arguments should be empty dict
            assert result.tool_calls[0].arguments == {}
