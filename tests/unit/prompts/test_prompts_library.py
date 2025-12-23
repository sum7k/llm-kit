from pathlib import Path

import pytest

from llm_kit.prompts.prompt import Prompt
from llm_kit.prompts.prompts_library import PromptsLibrary


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample YAML prompt files."""
    # First prompt
    (tmp_path / "greeting.yaml").write_text(
        """name: greeting
version: "1.0"
description: A simple greeting prompt
inputs:
  user_name: The name of the user to greet
template: Hello, {{ user_name }}!
"""
    )

    # Second prompt - different version of greeting
    (tmp_path / "greeting_v2.yaml").write_text(
        """name: greeting
version: "2.0"
description: An enhanced greeting prompt
inputs:
  user_name: The name of the user to greet
  time_of_day: Morning, afternoon, or evening
template: Good {{ time_of_day }}, {{ user_name }}!
"""
    )

    # Third prompt - different name
    (tmp_path / "summarize.yaml").write_text(
        """name: summarize
version: "1.0"
description: Summarize text content
inputs:
  text: The text to summarize
  max_length: Maximum length of summary
template: |
  Please summarize the following text in {{ max_length }} words or less:
  {{ text }}
"""
    )

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Create an empty temp directory."""
    return tmp_path


class TestPromptsLibrary:
    def test_loads_prompts_from_directory(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        assert len(library.list()) == 3

    def test_get_prompt_by_name_and_version(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        prompt = library.get("greeting", "1.0")

        assert prompt.name == "greeting"
        assert prompt.version == "1.0"
        assert prompt.description == "A simple greeting prompt"
        assert prompt.inputs == {"user_name": "The name of the user to greet"}
        assert prompt.template == "Hello, {{ user_name }}!"

    def test_get_different_versions_of_same_prompt(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        v1 = library.get("greeting", "1.0")
        v2 = library.get("greeting", "2.0")

        assert v1.version == "1.0"
        assert v2.version == "2.0"
        assert len(v1.inputs) == 1
        assert len(v2.inputs) == 2

    def test_get_raises_keyerror_for_unknown_prompt(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        with pytest.raises(KeyError, match="Prompt 'unknown' version '1.0' not found"):
            library.get("unknown", "1.0")

    def test_get_raises_keyerror_for_unknown_version(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        with pytest.raises(KeyError, match="Prompt 'greeting' version '9.9' not found"):
            library.get("greeting", "9.9")

    def test_list_returns_name_version_tuples(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        prompts = library.list()

        assert ("greeting", "1.0") in prompts
        assert ("greeting", "2.0") in prompts
        assert ("summarize", "1.0") in prompts

    def test_empty_directory_loads_no_prompts(self, empty_dir: Path) -> None:
        library = PromptsLibrary(str(empty_dir))

        assert library.list() == []

    def test_prompt_is_pydantic_model(self, prompts_dir: Path) -> None:
        library = PromptsLibrary(str(prompts_dir))

        prompt = library.get("summarize", "1.0")

        assert isinstance(prompt, Prompt)
