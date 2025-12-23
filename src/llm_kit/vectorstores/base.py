from collections.abc import Iterable
from typing import Protocol

from llm_kit.observability.base import MetricsHook

from .types import QueryResult, VectorItem


class VectorStore(Protocol):
    metrics_hook: MetricsHook

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

    def delete(
        self,
        *,
        namespace: str,
        ids: Iterable[str] | None = None,
        filters: dict | None = None,
    ) -> int:
        """
        Delete vectors by id or by metadata filter.
        Returns number of rows deleted.
        """
        ...
