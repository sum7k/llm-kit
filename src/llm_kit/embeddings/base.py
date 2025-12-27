from dataclasses import dataclass
from typing import Protocol

from llm_kit.observability.base import MetricsHook


@dataclass(frozen=True)
class Embedding:
    vector: list[float]


class EmbeddingsClient(Protocol):
    metrics_hook: MetricsHook

    async def embed(self, texts: list[str]) -> list[Embedding]: ...
