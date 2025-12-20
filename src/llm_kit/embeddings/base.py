from typing import Protocol


class EmbeddingsClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
