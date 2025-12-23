import time
from collections.abc import Generator

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

from llm_kit.vectorstores.pgvectorstore import PgVectorStore
from llm_kit.vectorstores.types import VectorItem


@pytest.fixture(scope="session")
def pg_dsn() -> Generator[str, None, None]:
    with PostgresContainer("pgvector/pgvector:pg16", driver=None) as pg:
        dsn = pg.get_connection_url()
        # Wait for postgres to be truly ready
        for _ in range(30):
            try:
                with psycopg.connect(dsn) as conn:
                    conn.execute("SELECT 1")
                    break
            except psycopg.OperationalError:
                time.sleep(0.5)
        yield dsn


@pytest.fixture(scope="session")
def _init_schema(pg_dsn: str) -> None:
    with psycopg.connect(pg_dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute("""
            CREATE TABLE vector_items (
                namespace TEXT NOT NULL,
                id TEXT NOT NULL,
                embedding VECTOR(3) NOT NULL,
                metadata JSONB NOT NULL,
                PRIMARY KEY (namespace, id)
            );
        """)
        conn.commit()


@pytest.fixture
def store(pg_dsn: str, _init_schema: None) -> Generator[PgVectorStore, None, None]:
    store = PgVectorStore(pg_dsn)
    yield store
    # Clean up after each test
    with psycopg.connect(pg_dsn) as conn:
        conn.execute("DELETE FROM vector_items;")
        conn.commit()


# --- Upsert ---


def test_upsert_inserts_new_rows(store: PgVectorStore) -> None:
    store.upsert(items=[VectorItem(id="a", vector=[1, 0, 0], metadata={"k": "v"})])

    results = store.query(vector=[1, 0, 0], top_k=10)
    assert len(results) == 1
    assert results[0].id == "a"


def test_upsert_updates_existing_rows(store: PgVectorStore) -> None:
    store.upsert(items=[VectorItem(id="a", vector=[1, 0, 0], metadata={"k": "old"})])
    store.upsert(items=[VectorItem(id="a", vector=[0, 1, 0], metadata={"k": "new"})])

    results = store.query(vector=[0, 1, 0], top_k=10)
    assert len(results) == 1
    assert results[0].metadata["k"] == "new"


# --- Query ---


def test_query_returns_nearest_neighbors(store: PgVectorStore) -> None:
    store.upsert(
        items=[
            VectorItem(id="close", vector=[1, 0, 0], metadata={}),
            VectorItem(id="far", vector=[0, 1, 0], metadata={}),
        ]
    )

    results = store.query(vector=[1, 0, 0], top_k=1)
    assert results[0].id == "close"


def test_query_respects_namespace_isolation(store: PgVectorStore) -> None:
    store.upsert(
        namespace="ns1", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )
    store.upsert(
        namespace="ns2", items=[VectorItem(id="b", vector=[1, 0, 0], metadata={})]
    )

    results = store.query(namespace="ns1", vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["a"]


def test_query_respects_metadata_filters(store: PgVectorStore) -> None:
    store.upsert(
        items=[
            VectorItem(id="match", vector=[1, 0, 0], metadata={"type": "x"}),
            VectorItem(id="nomatch", vector=[1, 0, 0], metadata={"type": "y"}),
        ]
    )

    results = store.query(vector=[1, 0, 0], top_k=10, filters={"type": "x"})
    assert [r.id for r in results] == ["match"]


# --- Delete ---


def test_delete_by_id(store: PgVectorStore) -> None:
    store.upsert(
        items=[
            VectorItem(id="a", vector=[1, 0, 0], metadata={}),
            VectorItem(id="b", vector=[1, 0, 0], metadata={}),
        ]
    )

    deleted = store.delete(ids=["a"])

    assert deleted == 1
    results = store.query(vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["b"]


def test_delete_by_filter(store: PgVectorStore) -> None:
    store.upsert(
        items=[
            VectorItem(id="a", vector=[1, 0, 0], metadata={"del": "yes"}),
            VectorItem(id="b", vector=[1, 0, 0], metadata={"del": "no"}),
        ]
    )

    deleted = store.delete(filters={"del": "yes"})

    assert deleted == 1
    results = store.query(vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["b"]


def test_delete_never_deletes_outside_namespace(store: PgVectorStore) -> None:
    store.upsert(
        namespace="ns1", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )
    store.upsert(
        namespace="ns2", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )

    store.delete(namespace="ns1", ids=["a"])

    results = store.query(namespace="ns2", vector=[1, 0, 0], top_k=10)
    assert len(results) == 1


# --- Score semantics ---


def test_higher_score_means_closer(store: PgVectorStore) -> None:
    store.upsert(
        items=[
            VectorItem(id="close", vector=[1, 0, 0], metadata={}),
            VectorItem(id="far", vector=[0, 1, 0], metadata={}),
        ]
    )

    results = store.query(vector=[1, 0, 0], top_k=2)
    assert results[0].score > results[1].score


# --- Empty results ---


def test_query_on_empty_namespace_returns_empty_list(store: PgVectorStore) -> None:
    results = store.query(namespace="nonexistent", vector=[1, 0, 0], top_k=10)
    assert results == []
