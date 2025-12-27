"""SQLite vector store using sqlite-vec extension for native vector operations."""

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path
from time import monotonic
from typing import Any

import apsw
import sqlite_vec

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import VectorStore
from .types import QueryResult, VectorItem

DEFAULT_NAMESPACE = "__global__"


class SQLiteVectorStore(VectorStore):
    """Vector store implementation using SQLite with sqlite-vec extension.

    Uses the vec0 virtual table for efficient vector storage and KNN queries.
    Supports cosine distance metric and metadata filtering.

    Requires the `apsw` package for full SQLite extension support.

    Example:
        >>> store = SQLiteVectorStore(db_path="vectors.db", dimensions=1536)
        >>> await store.upsert(items=[
        ...     VectorItem(id="1", vector=[0.1, 0.2, 0.3], metadata={"type": "doc"})
        ... ])
        >>> results = await store.query(vector=[0.1, 0.2, 0.3], top_k=5)
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        dimensions: int = 1536,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ) -> None:
        """
        Initialize SQLite vector store with sqlite-vec extension.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory database.
            dimensions: Dimension of vectors to store. Must be specified at table creation.
            metrics_hook: Hook for recording metrics.
        """
        self.metrics_hook = metrics_hook
        self._db_path = str(db_path)
        self._dimensions = dimensions
        self._conn: apsw.Connection | None = None

    def _get_connection(self) -> apsw.Connection:
        """Get or create SQLite connection with sqlite-vec loaded (lazy initialization)."""
        if self._conn is None:
            self._conn = apsw.Connection(self._db_path)

            # Load sqlite-vec extension
            self._conn.enableloadextension(True)
            self._conn.loadextension(sqlite_vec.loadable_path())
            self._conn.enableloadextension(False)

            self._initialize_schema()
        return self._conn

    @staticmethod
    def _make_key(namespace: str, item_id: str) -> str:
        """Create a composite key from namespace and item_id."""
        return f"{namespace}:{item_id}"

    @staticmethod
    def _parse_key(key: str, namespace: str) -> str:
        """Extract item_id from composite key."""
        prefix = f"{namespace}:"
        if key.startswith(prefix):
            return key[len(prefix) :]
        return key

    def _initialize_schema(self) -> None:
        """Create vec0 virtual table if it doesn't exist."""
        if self._conn is None:
            return

        # Create vec0 virtual table for vector storage with cosine distance
        # Using namespace as partition key for efficient multi-tenant queries
        # The primary key is a composite of namespace:item_id to ensure uniqueness
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
                composite_id TEXT PRIMARY KEY,
                namespace TEXT PARTITION KEY,
                embedding float[{self._dimensions}] distance_metric=cosine,
                +metadata TEXT,
                +item_id TEXT
            )
            """
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def upsert(
        self, *, namespace: str = DEFAULT_NAMESPACE, items: Iterable[VectorItem]
    ) -> None:
        """
        Insert or update vectors.

        sqlite-vec doesn't support ON CONFLICT for vec0, so we delete then insert.

        Args:
            namespace: Logical namespace for multi-tenancy.
            items: Iterable of VectorItem to upsert.
        """
        start = monotonic()
        items_list = list(items)

        if not items_list:
            return

        def _upsert() -> None:
            conn = self._get_connection()

            # Delete existing items first (sqlite-vec doesn't support UPDATE)
            composite_ids = [self._make_key(namespace, item.id) for item in items_list]
            placeholders = ",".join("?" * len(composite_ids))
            conn.execute(
                f"""
                DELETE FROM vec_items
                WHERE namespace = ? AND composite_id IN ({placeholders})
                """,
                (namespace, *composite_ids),
            )

            # Insert new items using sqlite-vec serialization
            for item in items_list:
                conn.execute(
                    """
                    INSERT INTO vec_items(composite_id, namespace, embedding, metadata, item_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self._make_key(namespace, item.id),
                        namespace,
                        sqlite_vec.serialize_float32(item.vector),
                        json.dumps(dict(item.metadata)),
                        item.id,
                    ),
                )

        await asyncio.to_thread(_upsert)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.SQLITE_UPSERT_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.SQLITE_OPERATIONS_TOTAL, labels={"operation": "upsert"}
        )

    async def query(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """
        Query for similar vectors using cosine distance via sqlite-vec.

        Args:
            namespace: Logical namespace to search within.
            vector: Query vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (exact match, applied post-query).

        Returns:
            List of QueryResult sorted by similarity (highest first).
        """
        start = monotonic()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        def _query() -> list[QueryResult]:
            conn = self._get_connection()

            # Serialize query vector to sqlite-vec format
            query_blob = sqlite_vec.serialize_float32(vector)

            # Fetch more results if we have filters (post-filtering)
            fetch_k = top_k * 3 if filters else top_k

            # KNN query using sqlite-vec MATCH syntax with partition key filter
            # We select item_id (auxiliary column) not composite_id (primary key)
            rows = list(
                conn.execute(
                    """
                    SELECT
                        item_id,
                        distance,
                        metadata
                    FROM vec_items
                    WHERE embedding MATCH ?
                        AND k = ?
                        AND namespace = ?
                    """,
                    (query_blob, fetch_k, namespace),
                )
            )

            results: list[QueryResult] = []

            for row in rows:
                item_id, distance, metadata_json = row
                metadata = json.loads(metadata_json)

                # Apply metadata filters if provided
                if filters:
                    match = all(metadata.get(k) == v for k, v in filters.items())
                    if not match:
                        continue

                # Convert cosine distance to similarity score
                # cosine distance is 0 for identical, 2 for opposite
                # Convert to 0-1 score where 1 is most similar
                score = 1.0 - (distance / 2.0)

                results.append(
                    QueryResult(
                        id=item_id,
                        score=score,
                        metadata=metadata,
                    )
                )

                if len(results) >= top_k:
                    break

            return results

        results = await asyncio.to_thread(_query)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.SQLITE_QUERY_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.SQLITE_OPERATIONS_TOTAL, labels={"operation": "query"}
        )

        return results

    async def delete(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        ids: Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """
        Delete vectors by id or by metadata filter.

        Args:
            namespace: Logical namespace.
            ids: Optional iterable of IDs to delete.
            filters: Optional metadata filters for deletion.

        Returns:
            Number of vectors deleted.
        """
        start = monotonic()
        if not ids and not filters:
            raise ValueError("delete requires ids or filters")

        def _delete() -> int:
            conn = self._get_connection()

            if ids and not filters:
                # Direct delete by IDs using composite keys
                id_list = list(ids)
                composite_ids = [self._make_key(namespace, id_) for id_ in id_list]
                placeholders = ",".join("?" * len(composite_ids))

                # Count existing
                count_before: int = list(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) FROM vec_items
                        WHERE namespace = ? AND composite_id IN ({placeholders})
                        """,
                        (namespace, *composite_ids),
                    )
                )[0][0]

                conn.execute(
                    f"""
                    DELETE FROM vec_items
                    WHERE namespace = ? AND composite_id IN ({placeholders})
                    """,
                    (namespace, *composite_ids),
                )
                return count_before
            else:
                # Need to query for metadata filtering
                rows = list(
                    conn.execute(
                        """
                        SELECT item_id, metadata
                        FROM vec_items
                        WHERE namespace = ?
                        """,
                        (namespace,),
                    )
                )

                # Filter by metadata and optionally by IDs
                id_set = set(ids) if ids else None
                matching_ids = []
                for row in rows:
                    item_id, metadata_json = row
                    if id_set and item_id not in id_set:
                        continue
                    if filters:
                        metadata = json.loads(metadata_json)
                        if not all(metadata.get(k) == v for k, v in filters.items()):
                            continue
                    matching_ids.append(item_id)

                if not matching_ids:
                    return 0

                # Delete matching IDs using composite keys
                composite_ids = [self._make_key(namespace, id_) for id_ in matching_ids]
                placeholders = ",".join("?" * len(composite_ids))
                conn.execute(
                    f"""
                    DELETE FROM vec_items
                    WHERE namespace = ? AND composite_id IN ({placeholders})
                    """,
                    (namespace, *composite_ids),
                )
                return len(matching_ids)

        deleted = await asyncio.to_thread(_delete)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.SQLITE_DELETE_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.SQLITE_OPERATIONS_TOTAL, labels={"operation": "delete"}
        )

        return deleted

    async def get_by_ids(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        ids: Iterable[str],
    ) -> list[VectorItem]:
        """
        Retrieve vectors by their IDs.

        Args:
            namespace: Logical namespace.
            ids: Iterable of IDs to retrieve.

        Returns:
            List of VectorItem for found IDs.
        """
        id_list = list(ids)
        if not id_list:
            return []

        def _get() -> list[VectorItem]:
            conn = self._get_connection()
            # Use composite keys for lookup
            composite_ids = [self._make_key(namespace, id_) for id_ in id_list]
            placeholders = ",".join("?" * len(composite_ids))
            rows = list(
                conn.execute(
                    f"""
                    SELECT item_id, embedding, metadata
                    FROM vec_items
                    WHERE namespace = ? AND composite_id IN ({placeholders})
                    """,
                    (namespace, *composite_ids),
                )
            )

            items = []
            for row in rows:
                item_id, embedding_blob, metadata_json = row
                # Deserialize vector from sqlite-vec blob using vec_to_json
                vec_json = list(
                    conn.execute("SELECT vec_to_json(?)", (embedding_blob,))
                )[0][0]
                vector = json.loads(vec_json)
                metadata = json.loads(metadata_json)
                items.append(
                    VectorItem(
                        id=item_id,
                        vector=vector,
                        metadata=metadata,
                    )
                )
            return items

        return await asyncio.to_thread(_get)

    async def count(self, *, namespace: str = DEFAULT_NAMESPACE) -> int:
        """
        Count vectors in a namespace.

        Args:
            namespace: Logical namespace.

        Returns:
            Number of vectors in the namespace.
        """

        def _count() -> int:
            conn = self._get_connection()
            result = list(
                conn.execute(
                    "SELECT COUNT(*) FROM vec_items WHERE namespace = ?",
                    (namespace,),
                )
            )
            count: int = result[0][0]
            return count

        return await asyncio.to_thread(_count)
