# src/llm_kit/embeddings/config.py

from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "local"]


@dataclass(frozen=True)
class EmbeddingsConfig:
    provider: Provider
    model: str
    timeout: float = 30.0
    batch_size: int = 100

    # provider-specific (used only when relevant)
    api_key: str | None = None
