import logging
from pathlib import Path

import yaml

from .prompt import Prompt

logger = logging.getLogger(__name__)


class PromptsLibrary:
    def __init__(self, directory: str) -> None:
        self._prompts: dict[tuple[str, str], Prompt] = {}
        logger.info("Initializing PromptsLibrary from directory: %s", directory)
        self._load_all(Path(directory))
        logger.info("Loaded %d prompts", len(self._prompts))

    def get(self, name: str, version: str) -> Prompt:
        logger.debug("Getting prompt: name=%s, version=%s", name, version)
        try:
            return self._prompts[(name, version)]
        except KeyError:
            logger.error("Prompt not found: name=%s, version=%s", name, version)
            raise KeyError(f"Prompt '{name}' version '{version}' not found")

    def list(self) -> list[tuple[str, str]]:
        return list(self._prompts.keys())

    def _load_all(self, directory: Path) -> None:
        for file_path in directory.glob("*.yaml"):
            prompt = self._load_prompt(file_path)
            self._prompts[(prompt.name, prompt.version)] = prompt
            logger.debug(
                "Loaded prompt: %s v%s from %s", prompt.name, prompt.version, file_path
            )

    def _load_prompt(self, file_path: Path) -> Prompt:
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return Prompt(**data)
