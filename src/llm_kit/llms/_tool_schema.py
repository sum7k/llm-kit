# src/llm_kit/llms/_tool_schema.py

"""Internal module for converting Tool definitions to provider-specific schemas.

This is infrastructure, not behavior. Pure data transformation.
"""

from llm_kit.tools.tool import Tool


def tools_to_openai_schema(tools: list[Tool]) -> list[dict]:
    """Convert Tool definitions to OpenAI function calling format.

    Args:
        tools: List of Tool objects with Pydantic input schemas.

    Returns:
        List of dicts in OpenAI's tool format.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema.model_json_schema(),
            },
        }
        for tool in tools
    ]


def tools_to_anthropic_schema(tools: list[Tool]) -> list[dict]:
    """Convert Tool definitions to Anthropic tool use format.

    Args:
        tools: List of Tool objects with Pydantic input schemas.

    Returns:
        List of dicts in Anthropic's tool format.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema.model_json_schema(),
        }
        for tool in tools
    ]
