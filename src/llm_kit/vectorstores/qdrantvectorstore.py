from collections.abc import Iterable
from time import monotonic
from typing import TypeAlias

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    HasVectorCondition,
    IsEmptyCondition,
    IsNullCondition,
    MatchValue,
    NestedCondition,
    PointStruct,
    VectorParams,
)

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import VectorStore
from .types import QueryResult, VectorItem

# Type alias for Qdrant filter conditions
Condition: TypeAlias = (
    FieldCondition
    | IsEmptyCondition
    | IsNullCondition
    | HasIdCondition
    | HasVectorCondition
    | NestedCondition
    | Filter
)

DEFAULT_NAMESPACE = "__global__"


class QdrantVectorStore(VectorStore):
    """Vector store implementation using Qdrant."""

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk: bool = False,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ) -> None:
        """
        Initialize Qdrant vector store.

        Args:
            url: Qdrant server URL. If provided, connects to remote server.
            path: Path to local Qdrant storage directory. If provided, uses local persistence.
            api_key: API key for Qdrant Cloud (only used with url).
            collection_name: Name of the collection.
            vector_size: Dimensionality of vectors.
            distance: Distance metric (COSINE, EUCLID, DOT).
            on_disk: Whether to store vectors on disk (for large datasets).
            metrics_hook: Hook for recording metrics.

        Note:
            - If both url and path are None, uses in-memory mode.
            - If url is provided, connects to remote Qdrant server.
            - If path is provided (and url is None), uses local file persistence.
            - Local storage (path) requires point IDs to be valid UUIDs.

        Examples:
            # In-memory (for testing)
            store = QdrantVectorStore(collection_name="test", vector_size=384)

            # Local file persistence (requires UUID IDs)
            store = QdrantVectorStore(path="./qdrant_data", collection_name="docs", vector_size=384)

            # Remote server
            store = QdrantVectorStore(url="http://localhost:6333", collection_name="docs", vector_size=384)
        """
        self.metrics_hook = metrics_hook

        if url:
            # Remote Qdrant server
            self._client = AsyncQdrantClient(url=url, api_key=api_key)
        elif path:
            # Local file persistence
            self._client = AsyncQdrantClient(path=path)
        else:
            # In-memory mode
            self._client = AsyncQdrantClient(":memory:")

        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance
        self._on_disk = on_disk

    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    on_disk=self._on_disk,
                ),
            )

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()

    async def upsert(
        self, *, namespace: str = DEFAULT_NAMESPACE, items: Iterable[VectorItem]
    ) -> None:
        """
        Insert or update vectors.

        Args:
            namespace: Logical namespace for multi-tenancy.
            items: Iterable of VectorItem to upsert.
        """
        await self._ensure_collection()

        start = monotonic()
        points = [
            PointStruct(
                id=item.id,
                vector=item.vector,
                payload={"_namespace": namespace, **dict(item.metadata)},
            )
            for item in items
        ]

        if not points:
            return

        await self._client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=points,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.QDRANT_UPSERT_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.QDRANT_OPERATIONS_TOTAL, labels={"operation": "upsert"}
        )

    async def query(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[QueryResult]:
        """
        Query for similar vectors.

        Args:
            namespace: Logical namespace to search within.
            vector: Query vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (exact match).

        Returns:
            List of QueryResult sorted by similarity (highest first).
        """
        await self._ensure_collection()

        start = monotonic()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        must_conditions: list[Condition] = [
            FieldCondition(key="_namespace", match=MatchValue(value=namespace))
        ]

        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        query_filter = Filter(must=must_conditions)

        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.QDRANT_QUERY_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.QDRANT_OPERATIONS_TOTAL, labels={"operation": "query"}
        )

        return [
            QueryResult(
                id=str(hit.id),
                score=hit.score,
                metadata={
                    k: v for k, v in (hit.payload or {}).items() if k != "_namespace"
                },
            )
            for hit in results.points
        ]

    async def delete(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        ids: Iterable[str] | None = None,
        filters: dict | None = None,
    ) -> int:
        """
        Delete vectors by id or by metadata filter.

        Args:
            namespace: Logical namespace.
            ids: Optional iterable of IDs to delete.
            filters: Optional metadata filters for deletion.

        Returns:
            Number of points deleted.
        """
        await self._ensure_collection()

        start = monotonic()
        if not ids and not filters:
            raise ValueError("delete requires ids or filters")

        # Count before deletion to return deleted count
        count_before = await self._count_matching(
            namespace=namespace, ids=ids, filters=filters
        )

        # Build filter conditions - always include namespace
        must_conditions: list[Condition] = [
            FieldCondition(key="_namespace", match=MatchValue(value=namespace))
        ]

        if ids:
            # Add HasIdCondition to filter by specific IDs
            must_conditions.append(HasIdCondition(has_id=list(ids)))

        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(filter=Filter(must=must_conditions)),
            wait=True,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.QDRANT_DELETE_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.QDRANT_OPERATIONS_TOTAL, labels={"operation": "delete"}
        )

        return count_before

    async def _count_matching(
        self,
        *,
        namespace: str,
        ids: Iterable[str] | None = None,
        filters: dict | None = None,
    ) -> int:
        """Count points matching the given criteria."""
        must_conditions: list[Condition] = [
            FieldCondition(key="_namespace", match=MatchValue(value=namespace))
        ]

        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        query_filter = Filter(must=must_conditions)

        if ids:
            # For ID-based deletion, we need to check which IDs exist
            id_list = list(ids)
            points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=id_list,
                with_payload=True,
            )
            # Filter by namespace
            return sum(
                1
                for p in points
                if p.payload and p.payload.get("_namespace") == namespace
            )
        else:
            result = await self._client.count(
                collection_name=self._collection_name,
                count_filter=query_filter,
                exact=True,
            )
            return int(result.count)
