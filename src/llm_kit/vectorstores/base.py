from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VectorItem:
    id: str
    vector: list[float]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class QueryResult:
    id: str
    score: float
    metadata: Mapping[str, Any]


class VectorStore(Protocol):
    def upsert(
        self,
        *,
        namespace: str,
        items: Iterable[VectorItem],
    ) -> None: ...

    def query(
        self,
        *,
        namespace: str,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[QueryResult]: ...
