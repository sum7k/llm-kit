import pytest
from pydantic import BaseModel

from llm_kit.tools.tool import Tool
from llm_kit.tools.tool_registry import ToolRegistry


class DummyInput(BaseModel):
    x: int


def dummy_handler(args: DummyInput) -> int:
    return args.x * 2


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def sample_tool() -> Tool:
    return Tool(
        name="double",
        description="Doubles a number",
        input_schema=DummyInput,
        handler=dummy_handler,
    )


def test_register_and_get(registry: ToolRegistry, sample_tool: Tool) -> None:
    registry.register(sample_tool)
    assert registry.get("double") is sample_tool


def test_register_duplicate_raises(registry: ToolRegistry, sample_tool: Tool) -> None:
    registry.register(sample_tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(sample_tool)


def test_get_unknown_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


def test_remove(registry: ToolRegistry, sample_tool: Tool) -> None:
    registry.register(sample_tool)
    registry.remove("double")
    with pytest.raises(KeyError):
        registry.get("double")


def test_remove_unknown_raises(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError, match="not found"):
        registry.remove("nonexistent")


def test_list_returns_copy(registry: ToolRegistry, sample_tool: Tool) -> None:
    registry.register(sample_tool)
    tools = registry.list()
    tools.clear()  # mutate the copy
    assert registry.get("double") is sample_tool  # original unaffected
