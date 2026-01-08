"""Integration tests for QdrantVectorStore using local Qdrant instance."""

import contextlib
import os
import uuid
from collections.abc import AsyncGenerator

import pytest

from llm_kit.vectorstores.qdrantvectorstore import QdrantVectorStore
from llm_kit.vectorstores.types import VectorItem

VECTOR_SIZE = 4
COLLECTION_NAME = "llm-kit-test"


def make_id(name: str) -> str:
    """Generate a deterministic UUID from a name for test reproducibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    """Return the Qdrant connection URL (local instance on port 6333)."""
    return os.environ.get("QDRANT_URL", "http://localhost:6333")


@pytest.fixture
async def store(qdrant_url: str) -> AsyncGenerator[QdrantVectorStore, None]:
    """Create a fresh QdrantVectorStore for each test."""
    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=COLLECTION_NAME,
        vector_size=VECTOR_SIZE,
    )

    # Delete collection if it exists to start fresh
    try:
        await store._client.delete_collection(COLLECTION_NAME)
    except Exception:
        contextlib.suppress(Exception)
    yield store

    # Cleanup: delete collection after test
    try:
        await store._client.delete_collection(COLLECTION_NAME)
    except Exception:
        contextlib.suppress(Exception)
    await store.close()


# =============================================================================
# Test Cases
# =============================================================================

# Pre-generate UUIDs for test items
ID_A = make_id("a")
ID_B = make_id("b")
ID_C = make_id("c")
ID_X = make_id("x")
ID_SAME = make_id("same")
ID_DIFF = make_id("diff")


@pytest.mark.asyncio
async def test_upsert_and_query_basic(store: QdrantVectorStore) -> None:
    """Test basic upsert and query functionality."""
    items = [
        VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={"type": "alpha"}),
        VectorItem(id=ID_B, vector=[0.0, 1.0, 0.0, 0.0], metadata={"type": "beta"}),
    ]
    await store.upsert(items=items)

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].id == ID_A
    assert results[0].metadata["type"] == "alpha"


@pytest.mark.asyncio
async def test_upsert_updates_existing(store: QdrantVectorStore) -> None:
    """Test that upsert updates existing items."""
    await store.upsert(
        items=[VectorItem(id=ID_X, vector=[1.0, 0.0, 0.0, 0.0], metadata={"v": 1})]
    )
    await store.upsert(
        items=[VectorItem(id=ID_X, vector=[0.0, 1.0, 0.0, 0.0], metadata={"v": 2})]
    )

    results = await store.query(vector=[0.0, 1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].id == ID_X
    assert results[0].metadata["v"] == 2


@pytest.mark.asyncio
async def test_query_with_filters(store: QdrantVectorStore) -> None:
    """Test querying with metadata filters."""
    items = [
        VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={"color": "red"}),
        VectorItem(id=ID_B, vector=[0.9, 0.1, 0.0, 0.0], metadata={"color": "blue"}),
        VectorItem(id=ID_C, vector=[0.8, 0.2, 0.0, 0.0], metadata={"color": "red"}),
    ]
    await store.upsert(items=items)

    results = await store.query(
        vector=[1.0, 0.0, 0.0, 0.0], top_k=10, filters={"color": "red"}
    )

    assert len(results) == 2
    assert all(r.metadata["color"] == "red" for r in results)


@pytest.mark.asyncio
async def test_query_respects_top_k(store: QdrantVectorStore) -> None:
    """Test that query respects top_k limit."""
    items = [
        VectorItem(id=make_id(str(i)), vector=[float(i), 0.0, 0.0, 0.0], metadata={})
        for i in range(10)
    ]
    await store.upsert(items=items)

    results = await store.query(vector=[5.0, 0.0, 0.0, 0.0], top_k=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_query_top_k_validation(store: QdrantVectorStore) -> None:
    """Test that top_k < 1 raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=0)


@pytest.mark.asyncio
async def test_delete_by_ids(store: QdrantVectorStore) -> None:
    """Test deleting items by IDs."""
    items = [
        VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={}),
        VectorItem(id=ID_B, vector=[0.0, 1.0, 0.0, 0.0], metadata={}),
        VectorItem(id=ID_C, vector=[0.0, 0.0, 1.0, 0.0], metadata={}),
    ]
    await store.upsert(items=items)

    deleted = await store.delete(ids=[ID_A, ID_B])

    assert deleted == 2

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].id == ID_C


@pytest.mark.asyncio
async def test_delete_by_filters(store: QdrantVectorStore) -> None:
    """Test deleting items by metadata filters."""
    items = [
        VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={"status": "active"}),
        VectorItem(
            id=ID_B, vector=[0.0, 1.0, 0.0, 0.0], metadata={"status": "inactive"}
        ),
        VectorItem(id=ID_C, vector=[0.0, 0.0, 1.0, 0.0], metadata={"status": "active"}),
    ]
    await store.upsert(items=items)

    deleted = await store.delete(filters={"status": "active"})

    assert deleted == 2

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].id == ID_B


@pytest.mark.asyncio
async def test_delete_requires_ids_or_filters(store: QdrantVectorStore) -> None:
    """Test that delete raises ValueError without ids or filters."""
    with pytest.raises(ValueError, match="delete requires ids or filters"):
        await store.delete()


@pytest.mark.asyncio
async def test_namespace_isolation(store: QdrantVectorStore) -> None:
    """Test that namespaces isolate data."""
    await store.upsert(
        namespace="ns1",
        items=[VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={"ns": "1"})],
    )
    await store.upsert(
        namespace="ns2",
        items=[VectorItem(id=ID_B, vector=[0.0, 1.0, 0.0, 0.0], metadata={"ns": "2"})],
    )

    results_ns1 = await store.query(
        namespace="ns1", vector=[1.0, 0.0, 0.0, 0.0], top_k=10
    )
    results_ns2 = await store.query(
        namespace="ns2", vector=[1.0, 0.0, 0.0, 0.0], top_k=10
    )

    assert len(results_ns1) == 1
    assert results_ns1[0].metadata["ns"] == "1"

    assert len(results_ns2) == 1
    assert results_ns2[0].metadata["ns"] == "2"


@pytest.mark.asyncio
async def test_query_empty_collection(store: QdrantVectorStore) -> None:
    """Test querying an empty collection returns empty list."""
    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)

    assert results == []


@pytest.mark.asyncio
async def test_score_semantics_cosine(store: QdrantVectorStore) -> None:
    """Test that scores follow cosine similarity semantics (1.0 = identical)."""
    items = [
        VectorItem(id=ID_SAME, vector=[1.0, 0.0, 0.0, 0.0], metadata={}),
        VectorItem(id=ID_DIFF, vector=[0.0, 1.0, 0.0, 0.0], metadata={}),
    ]
    await store.upsert(items=items)

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=2)

    # Identical vector should have score close to 1.0
    same_result = next(r for r in results if r.id == ID_SAME)
    assert same_result.score > 0.99

    # Orthogonal vector should have score close to 0.0
    diff_result = next(r for r in results if r.id == ID_DIFF)
    assert diff_result.score < 0.01


@pytest.mark.asyncio
async def test_upsert_empty_items(store: QdrantVectorStore) -> None:
    """Test that upserting empty items is a no-op."""
    await store.upsert(items=[])

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_in_memory_mode() -> None:
    """Test that in-memory mode works without a server."""
    store = QdrantVectorStore(
        collection_name="test_memory",
        vector_size=VECTOR_SIZE,
    )

    items = [
        VectorItem(id=ID_A, vector=[1.0, 0.0, 0.0, 0.0], metadata={"key": "value"}),
    ]
    await store.upsert(items=items)

    results = await store.query(vector=[1.0, 0.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].id == ID_A
    assert results[0].metadata["key"] == "value"

    await store.close()
