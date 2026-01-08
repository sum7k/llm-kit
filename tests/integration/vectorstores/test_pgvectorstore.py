import os
from collections.abc import AsyncGenerator

import psycopg
import pytest

from llm_kit.vectorstores.pgvectorstore import PgVectorStore
from llm_kit.vectorstores.types import VectorItem


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Use local PostgreSQL with pgvector extension.

    Set PG_DSN env var to override. Default uses localhost:5432.
    """
    return os.environ.get(
        "PG_DSN", "postgresql://postgres:postgres@localhost:5432/llm-kit-test"
    )


@pytest.fixture(scope="session")
def _init_schema(pg_dsn: str) -> None:
    # Connect to default postgres db to create test database
    admin_dsn = pg_dsn.rsplit("/", 1)[0] + "/postgres"
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        # Check if database exists
        result = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'llm-kit-test'"
        ).fetchone()
        if not result:
            conn.execute('CREATE DATABASE "llm-kit-test"')

    # Now connect to test db and create schema
    with psycopg.connect(pg_dsn) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute("DROP TABLE IF EXISTS vector_items;")
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
async def store(pg_dsn: str, _init_schema: None) -> AsyncGenerator[PgVectorStore, None]:
    store = PgVectorStore(pg_dsn)
    await store._pool.open()
    yield store
    # Clean up after each test
    with psycopg.connect(pg_dsn) as conn:
        conn.execute("DELETE FROM vector_items;")
        conn.commit()
    await store.close()


# --- Upsert ---


@pytest.mark.asyncio
async def test_upsert_inserts_new_rows(store: PgVectorStore) -> None:
    await store.upsert(
        items=[VectorItem(id="a", vector=[1, 0, 0], metadata={"k": "v"})]
    )

    results = await store.query(vector=[1, 0, 0], top_k=10)
    assert len(results) == 1
    assert results[0].id == "a"


@pytest.mark.asyncio
async def test_upsert_updates_existing_rows(store: PgVectorStore) -> None:
    await store.upsert(
        items=[VectorItem(id="a", vector=[1, 0, 0], metadata={"k": "old"})]
    )
    await store.upsert(
        items=[VectorItem(id="a", vector=[0, 1, 0], metadata={"k": "new"})]
    )

    results = await store.query(vector=[0, 1, 0], top_k=10)
    assert len(results) == 1
    assert results[0].metadata["k"] == "new"


# --- Query ---


@pytest.mark.asyncio
async def test_query_returns_nearest_neighbors(store: PgVectorStore) -> None:
    await store.upsert(
        items=[
            VectorItem(id="close", vector=[1, 0, 0], metadata={}),
            VectorItem(id="far", vector=[0, 1, 0], metadata={}),
        ]
    )

    results = await store.query(vector=[1, 0, 0], top_k=1)
    assert results[0].id == "close"


@pytest.mark.asyncio
async def test_query_respects_namespace_isolation(store: PgVectorStore) -> None:
    await store.upsert(
        namespace="ns1", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )
    await store.upsert(
        namespace="ns2", items=[VectorItem(id="b", vector=[1, 0, 0], metadata={})]
    )

    results = await store.query(namespace="ns1", vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["a"]


@pytest.mark.asyncio
async def test_query_respects_metadata_filters(store: PgVectorStore) -> None:
    await store.upsert(
        items=[
            VectorItem(id="match", vector=[1, 0, 0], metadata={"type": "x"}),
            VectorItem(id="nomatch", vector=[1, 0, 0], metadata={"type": "y"}),
        ]
    )

    results = await store.query(vector=[1, 0, 0], top_k=10, filters={"type": "x"})
    assert [r.id for r in results] == ["match"]


# --- Delete ---


@pytest.mark.asyncio
async def test_delete_by_id(store: PgVectorStore) -> None:
    await store.upsert(
        items=[
            VectorItem(id="a", vector=[1, 0, 0], metadata={}),
            VectorItem(id="b", vector=[1, 0, 0], metadata={}),
        ]
    )

    deleted = await store.delete(ids=["a"])

    assert deleted == 1
    results = await store.query(vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["b"]


@pytest.mark.asyncio
async def test_delete_by_filter(store: PgVectorStore) -> None:
    await store.upsert(
        items=[
            VectorItem(id="a", vector=[1, 0, 0], metadata={"del": "yes"}),
            VectorItem(id="b", vector=[1, 0, 0], metadata={"del": "no"}),
        ]
    )

    deleted = await store.delete(filters={"del": "yes"})

    assert deleted == 1
    results = await store.query(vector=[1, 0, 0], top_k=10)
    assert [r.id for r in results] == ["b"]


@pytest.mark.asyncio
async def test_delete_never_deletes_outside_namespace(store: PgVectorStore) -> None:
    await store.upsert(
        namespace="ns1", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )
    await store.upsert(
        namespace="ns2", items=[VectorItem(id="a", vector=[1, 0, 0], metadata={})]
    )

    await store.delete(namespace="ns1", ids=["a"])

    results = await store.query(namespace="ns2", vector=[1, 0, 0], top_k=10)
    assert len(results) == 1


# --- Score semantics ---


@pytest.mark.asyncio
async def test_higher_score_means_closer(store: PgVectorStore) -> None:
    await store.upsert(
        items=[
            VectorItem(id="close", vector=[1, 0, 0], metadata={}),
            VectorItem(id="far", vector=[0, 1, 0], metadata={}),
        ]
    )

    results = await store.query(vector=[1, 0, 0], top_k=2)
    assert results[0].score > results[1].score


# --- Empty results ---


@pytest.mark.asyncio
async def test_query_on_empty_namespace_returns_empty_list(
    store: PgVectorStore,
) -> None:
    results = await store.query(namespace="nonexistent", vector=[1, 0, 0], top_k=10)
    assert results == []
