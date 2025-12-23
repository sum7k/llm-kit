from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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
