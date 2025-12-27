# tests/unit/llms/test_tool_schema.py

import pytest
from pydantic import BaseModel, Field

from llm_kit.llms._tool_schema import tools_to_anthropic_schema, tools_to_openai_schema
from llm_kit.tools.tool import Tool


class SearchInput(BaseModel):
    """Search for documents."""

    query: str = Field(description="The search query")
    limit: int = Field(default=10, description="Max results")


class WeatherInput(BaseModel):
    city: str


@pytest.fixture
def sample_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Search the knowledge base",
            input_schema=SearchInput,
            handler=lambda _: [],
        ),
        Tool(
            name="get_weather",
            description="Get current weather",
            input_schema=WeatherInput,
            handler=lambda _: "sunny",
        ),
    ]


class TestToolSchemaConversion:
    def test_tools_to_openai_schema(self, sample_tools: list[Tool]) -> None:
        """Test conversion to OpenAI format."""
        schema = tools_to_openai_schema(sample_tools)

        assert len(schema) == 2

        # Check first tool
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "search"
        assert schema[0]["function"]["description"] == "Search the knowledge base"
        assert "properties" in schema[0]["function"]["parameters"]
        assert "query" in schema[0]["function"]["parameters"]["properties"]

        # Check second tool
        assert schema[1]["function"]["name"] == "get_weather"

    def test_tools_to_anthropic_schema(self, sample_tools: list[Tool]) -> None:
        """Test conversion to Anthropic format."""
        schema = tools_to_anthropic_schema(sample_tools)

        assert len(schema) == 2

        # Check first tool (Anthropic format is flatter)
        assert schema[0]["name"] == "search"
        assert schema[0]["description"] == "Search the knowledge base"
        assert "properties" in schema[0]["input_schema"]

        # Check second tool
        assert schema[1]["name"] == "get_weather"

    def test_empty_tools_list(self) -> None:
        """Test empty tools list."""
        assert tools_to_openai_schema([]) == []
        assert tools_to_anthropic_schema([]) == []

    def test_schema_includes_field_descriptions(self, sample_tools: list[Tool]) -> None:
        """Test that field descriptions are included in schema."""
        schema = tools_to_openai_schema(sample_tools)

        query_prop = schema[0]["function"]["parameters"]["properties"]["query"]
        assert query_prop.get("description") == "The search query"
