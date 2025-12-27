# tests/unit/vectorstores/test_sqlitevectorstore.py

import tempfile
from pathlib import Path

import pytest

from llm_kit.vectorstores import SQLiteVectorStore, VectorItem


class TestSQLiteVectorStore:
    @pytest.mark.asyncio
    async def test_upsert_and_query(self) -> None:
        """Test basic upsert and query operations."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=3)

        # Upsert some vectors
        items = [
            VectorItem(id="1", vector=[1.0, 0.0, 0.0], metadata={"type": "doc"}),
            VectorItem(id="2", vector=[0.0, 1.0, 0.0], metadata={"type": "doc"}),
            VectorItem(id="3", vector=[0.0, 0.0, 1.0], metadata={"type": "other"}),
        ]
        await store.upsert(items=items)

        # Query for similar vectors
        results = await store.query(vector=[1.0, 0.0, 0.0], top_k=2)

        assert len(results) == 2
        assert results[0].id == "1"  # Most similar
        assert results[0].score > results[1].score

        await store.close()

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self) -> None:
        """Test that upsert updates existing vectors."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        # Insert initial vector
        await store.upsert(
            items=[VectorItem(id="1", vector=[1.0, 0.0], metadata={"version": "v1"})]
        )

        # Update with new vector and metadata
        await store.upsert(
            items=[VectorItem(id="1", vector=[0.0, 1.0], metadata={"version": "v2"})]
        )

        # Query should return updated vector
        results = await store.query(vector=[0.0, 1.0], top_k=1)
        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].metadata["version"] == "v2"

        await store.close()

    @pytest.mark.asyncio
    async def test_query_with_filters(self) -> None:
        """Test querying with metadata filters."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        items = [
            VectorItem(
                id="1", vector=[1.0, 0.0], metadata={"type": "doc", "lang": "en"}
            ),
            VectorItem(
                id="2", vector=[1.0, 0.1], metadata={"type": "doc", "lang": "es"}
            ),
            VectorItem(
                id="3", vector=[1.0, 0.2], metadata={"type": "other", "lang": "en"}
            ),
        ]
        await store.upsert(items=items)

        # Query with type filter
        results = await store.query(
            vector=[1.0, 0.0], top_k=10, filters={"type": "doc"}
        )
        assert len(results) == 2
        assert all(r.metadata["type"] == "doc" for r in results)

        # Query with multiple filters
        results = await store.query(
            vector=[1.0, 0.0], top_k=10, filters={"type": "doc", "lang": "en"}
        )
        assert len(results) == 1
        assert results[0].id == "1"

        await store.close()

    @pytest.mark.asyncio
    async def test_delete_by_ids(self) -> None:
        """Test deleting vectors by IDs."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        items = [
            VectorItem(id="1", vector=[1.0, 0.0], metadata={}),
            VectorItem(id="2", vector=[0.0, 1.0], metadata={}),
            VectorItem(id="3", vector=[1.0, 1.0], metadata={}),
        ]
        await store.upsert(items=items)

        # Delete by IDs
        deleted = await store.delete(ids=["1", "3"])
        assert deleted == 2

        # Verify only item 2 remains
        results = await store.query(vector=[1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].id == "2"

        await store.close()

    @pytest.mark.asyncio
    async def test_delete_by_filters(self) -> None:
        """Test deleting vectors by metadata filters."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        items = [
            VectorItem(id="1", vector=[1.0, 0.0], metadata={"type": "doc"}),
            VectorItem(id="2", vector=[0.0, 1.0], metadata={"type": "doc"}),
            VectorItem(id="3", vector=[1.0, 1.0], metadata={"type": "other"}),
        ]
        await store.upsert(items=items)

        # Delete by metadata filter
        deleted = await store.delete(filters={"type": "doc"})
        assert deleted == 2

        # Verify only "other" type remains
        results = await store.query(vector=[1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].metadata["type"] == "other"

        await store.close()

    @pytest.mark.asyncio
    async def test_namespaces(self) -> None:
        """Test namespace isolation."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        # Insert into different namespaces
        await store.upsert(
            namespace="ns1",
            items=[VectorItem(id="1", vector=[1.0, 0.0], metadata={})],
        )
        await store.upsert(
            namespace="ns2",
            items=[VectorItem(id="1", vector=[0.0, 1.0], metadata={})],
        )

        # Query in ns1
        results_ns1 = await store.query(namespace="ns1", vector=[1.0, 0.0], top_k=10)
        assert len(results_ns1) == 1
        assert results_ns1[0].id == "1"

        # Query in ns2
        results_ns2 = await store.query(namespace="ns2", vector=[0.0, 1.0], top_k=10)
        assert len(results_ns2) == 1
        assert results_ns2[0].id == "1"

        # Delete from ns1 shouldn't affect ns2
        await store.delete(namespace="ns1", ids=["1"])
        results_ns1 = await store.query(namespace="ns1", vector=[1.0, 0.0], top_k=10)
        results_ns2 = await store.query(namespace="ns2", vector=[0.0, 1.0], top_k=10)
        assert len(results_ns1) == 0
        assert len(results_ns2) == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_persistent_storage(self) -> None:
        """Test that vectors persist to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create store and insert data
            store1 = SQLiteVectorStore(db_path=db_path, dimensions=2)
            await store1.upsert(
                items=[VectorItem(id="1", vector=[1.0, 0.0], metadata={"test": "data"})]
            )
            await store1.close()

            # Reopen and verify data persists
            store2 = SQLiteVectorStore(db_path=db_path, dimensions=2)
            results = await store2.query(vector=[1.0, 0.0], top_k=10)
            assert len(results) == 1
            assert results[0].id == "1"
            assert results[0].metadata["test"] == "data"
            await store2.close()

    @pytest.mark.asyncio
    async def test_empty_upsert(self) -> None:
        """Test that empty upsert doesn't raise an error."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)
        await store.upsert(items=[])
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_requires_ids_or_filters(self) -> None:
        """Test that delete requires either ids or filters."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        with pytest.raises(ValueError, match="delete requires ids or filters"):
            await store.delete()

        await store.close()

    @pytest.mark.asyncio
    async def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation via sqlite-vec."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=3)

        # Identical vectors should have highest similarity
        items = [
            VectorItem(id="1", vector=[1.0, 0.0, 0.0], metadata={}),
            VectorItem(id="2", vector=[0.0, 1.0, 0.0], metadata={}),
            VectorItem(id="3", vector=[1.0, 0.0, 0.0], metadata={}),  # Same as 1
        ]
        await store.upsert(items=items)

        results = await store.query(vector=[1.0, 0.0, 0.0], top_k=3)

        # Items 1 and 3 should have same high score (exact match)
        assert results[0].score == results[1].score
        assert results[0].score > results[2].score

        await store.close()

    @pytest.mark.asyncio
    async def test_get_by_ids(self) -> None:
        """Test retrieving vectors by IDs."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=3)

        items = [
            VectorItem(id="1", vector=[1.0, 0.0, 0.0], metadata={"type": "a"}),
            VectorItem(id="2", vector=[0.0, 1.0, 0.0], metadata={"type": "b"}),
            VectorItem(id="3", vector=[0.0, 0.0, 1.0], metadata={"type": "c"}),
        ]
        await store.upsert(items=items)

        # Get specific IDs
        retrieved = await store.get_by_ids(ids=["1", "3"])
        assert len(retrieved) == 2

        ids = {item.id for item in retrieved}
        assert ids == {"1", "3"}

        await store.close()

    @pytest.mark.asyncio
    async def test_count(self) -> None:
        """Test counting vectors in namespace."""
        store = SQLiteVectorStore(db_path=":memory:", dimensions=2)

        # Initially empty
        count = await store.count()
        assert count == 0

        # Add some vectors
        items = [
            VectorItem(id="1", vector=[1.0, 0.0], metadata={}),
            VectorItem(id="2", vector=[0.0, 1.0], metadata={}),
        ]
        await store.upsert(items=items)

        count = await store.count()
        assert count == 2

        # Add to different namespace
        await store.upsert(
            namespace="other",
            items=[VectorItem(id="3", vector=[1.0, 1.0], metadata={})],
        )

        # Default namespace should still have 2
        assert await store.count() == 2
        assert await store.count(namespace="other") == 1

        await store.close()
