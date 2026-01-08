import pytest
from pydantic import BaseModel, ValidationError

from llm_kit.tools.tool import Tool, ToolCall
from llm_kit.tools.tool_engine import ToolEngine
from llm_kit.tools.tool_registry import ToolRegistry


class AddInput(BaseModel):
    a: int
    b: int


def add_handler(args: AddInput) -> int:
    return args.a + args.b


@pytest.fixture
def engine() -> ToolEngine:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="add",
            description="Adds two numbers",
            input_schema=AddInput,
            handler=add_handler,
        )
    )
    return ToolEngine(registry)


@pytest.mark.asyncio
async def test_call_tool_returns_result(engine: ToolEngine) -> None:
    result = await engine.call_tool(
        ToolCall(tool_name="add", arguments={"a": 2, "b": 3})
    )
    assert result == 5


@pytest.mark.asyncio
async def test_call_tool_validates_input(engine: ToolEngine) -> None:
    with pytest.raises(ValidationError):  # Pydantic ValidationError
        await engine.call_tool(
            ToolCall(tool_name="add", arguments={"a": "not_int", "b": 3})
        )


@pytest.mark.asyncio
async def test_call_unknown_tool_raises(engine: ToolEngine) -> None:
    with pytest.raises(KeyError, match="not found"):
        await engine.call_tool(ToolCall(tool_name="unknown", arguments={}))
