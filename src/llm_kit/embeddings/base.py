from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Embedding:
    vector: list[float]


class EmbeddingsClient(Protocol):
    def embed(self, texts: list[str]) -> list[Embedding]: ...
