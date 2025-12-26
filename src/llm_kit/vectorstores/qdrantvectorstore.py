from collections.abc import Iterable
from time import monotonic
from typing import TypeAlias

from qdrant_client import QdrantClient
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
            url: Qdrant server URL. If None, uses in-memory mode.
            api_key: API key for Qdrant Cloud.
            collection_name: Name of the collection.
            vector_size: Dimensionality of vectors.
            distance: Distance metric (COSINE, EUCLID, DOT).
            on_disk: Whether to store vectors on disk (for large datasets).
            metrics_hook: Hook for recording metrics.
        """
        self.metrics_hook = metrics_hook
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")

        self._collection_name = collection_name
        self._vector_size = vector_size
        self._distance = distance

        self._ensure_collection(on_disk=on_disk)

    def _ensure_collection(self, *, on_disk: bool) -> None:
        """Create collection if it doesn't exist."""
        if not self._client.collection_exists(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                    on_disk=on_disk,
                ),
            )

    def close(self) -> None:
        """Close the Qdrant client."""
        self._client.close()

    def upsert(
        self, *, namespace: str = DEFAULT_NAMESPACE, items: Iterable[VectorItem]
    ) -> None:
        """
        Insert or update vectors.

        Args:
            namespace: Logical namespace for multi-tenancy.
            items: Iterable of VectorItem to upsert.
        """
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

        self._client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=points,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="qdrant_upsert_duration", value_ms=elapsed_ms
        )

    def query(
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

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="qdrant_query_duration", value_ms=elapsed_ms
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

    def delete(
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
        start = monotonic()
        if not ids and not filters:
            raise ValueError("delete requires ids or filters")

        # Count before deletion to return deleted count
        count_before = self._count_matching(
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

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(filter=Filter(must=must_conditions)),
            wait=True,
        )

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="qdrant_delete_duration", value_ms=elapsed_ms
        )

        return count_before

    def _count_matching(
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
            points = self._client.retrieve(
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
            result = self._client.count(
                collection_name=self._collection_name,
                count_filter=query_filter,
                exact=True,
            )
            return int(result.count)
