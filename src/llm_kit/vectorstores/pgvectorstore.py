import os
from collections.abc import Iterable
from time import monotonic

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import VectorStore
from .types import QueryResult, VectorItem

DEFAULT_NAMESPACE = "__global__"


def _configure_connection(conn: psycopg.Connection) -> None:  # type: ignore[type-arg]
    """Register pgvector types on new connections."""
    register_vector(conn)


class PgVectorStore(VectorStore):
    def __init__(
        self,
        dsn: str,
        pool_min_size: int | None = None,
        pool_max_size: int | None = None,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ) -> None:
        self.metrics_hook = metrics_hook
        pool_min_size = self._get_param_value(
            pool_min_size, "LLM_KIT_PG_POOL_MIN_SIZE", 1
        )
        pool_max_size = self._get_param_value(
            pool_max_size, "LLM_KIT_PG_POOL_MAX_SIZE", 10
        )
        self._pool = ConnectionPool(
            dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            configure=_configure_connection,
        )

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()

    def upsert(
        self, *, namespace: str = DEFAULT_NAMESPACE, items: Iterable[VectorItem]
    ) -> None:
        start = monotonic()
        rows = [
            (
                namespace,
                item.id,
                np.array(item.vector),
                Json(dict(item.metadata)),
            )
            for item in items
        ]

        if not rows:
            return

        query = sql.SQL(
            """
        INSERT INTO vector_items (namespace, id, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (namespace, id)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata;
        """
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(query, rows)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="pgvector_upsert_duration", value_ms=elapsed_ms
        )

    def query(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[QueryResult]:
        start = monotonic()
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        where_clauses = [sql.SQL("namespace = %s")]
        params: list = [namespace]

        if filters:
            for key, value in filters.items():
                where_clauses.append(sql.SQL("metadata ->> %s = %s"))
                params.extend([key, str(value)])

        where_sql = sql.SQL(" AND ").join(where_clauses)

        query = sql.SQL(
            """
        SELECT
            id,
            1 - (embedding <=> %s) AS score,
            metadata
        FROM vector_items
        WHERE {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        ).format(where_clause=where_sql)

        vector_arr = np.array(vector)
        params = [vector_arr] + params + [vector_arr, top_k]

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="pgvector_query_duration", value_ms=elapsed_ms
        )

        return [
            QueryResult(
                id=row[0],
                score=row[1],
                metadata=row[2],
            )
            for row in rows
        ]

    def delete(
        self,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        ids: Iterable[str] | None = None,
        filters: dict | None = None,
    ) -> int:
        start = monotonic()
        if not ids and not filters:
            raise ValueError("delete requires ids or filters")

        where_clauses: list[sql.SQL] = [sql.SQL("namespace = %s")]
        params: list = [namespace]

        if ids:
            where_clauses.append(sql.SQL("id = ANY(%s)"))
            params.append(list(ids))

        if filters:
            for key, value in filters.items():
                where_clauses.append(sql.SQL("metadata ->> %s = %s"))
                params.extend([key, str(value)])

        where_sql = sql.SQL(" AND ").join(where_clauses)

        delete_query = sql.SQL(
            """
        DELETE FROM vector_items
        WHERE {where_clause};
        """
        ).format(where_clause=where_sql)

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(delete_query, params)
            deleted: int = cur.rowcount

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(
            name="pgvector_delete_duration", value_ms=elapsed_ms
        )

        return deleted

    @staticmethod
    def _get_param_value(passed_value: int | None, env_var: str, default: int) -> int:
        if passed_value is not None:
            return passed_value
        env_value = os.environ.get(env_var)
        if env_value is not None:
            return int(env_value)
        return default
