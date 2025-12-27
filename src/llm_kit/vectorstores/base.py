from collections.abc import Iterable
from typing import Protocol

from llm_kit.observability.base import MetricsHook

from .types import QueryResult, VectorItem


class VectorStore(Protocol):
    metrics_hook: MetricsHook

    async def upsert(
        self,
        *,
        namespace: str,
        items: Iterable[VectorItem],
    ) -> None: ...

    async def query(
        self,
        *,
        namespace: str,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[QueryResult]: ...

    async def delete(
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
