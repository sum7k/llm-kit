import pytest
from pydantic import ValidationError

from llm_kit.tools.tool import ToolCall


def test_tool_call_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolCall(tool_name="test", arguments={}, extra_field="bad")
