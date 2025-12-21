import pytest

from llm_kit.chunking.chunking import Chunk, chunk_text


class TestChunkText:
    def test_single_chunk_when_text_fits(self) -> None:
        """Text smaller than chunk_size produces one chunk."""
        result = chunk_text(
            "hello", chunk_size=10, overlap=0, metadata={"source_id": "doc1"}
        )

        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].offset_start == 0
        assert result[0].offset_end == 5

    def test_multiple_chunks_with_no_overlap(self) -> None:
        """Text is split into non-overlapping chunks."""
        result = chunk_text(
            "abcdefghij", chunk_size=4, overlap=0, metadata={"source_id": "doc1"}
        )

        assert [c.text for c in result] == ["abcd", "efgh", "ij"]
        assert [(c.offset_start, c.offset_end) for c in result] == [
            (0, 4),
            (4, 8),
            (8, 10),
        ]

    def test_overlap_creates_overlapping_chunks(self) -> None:
        """Overlap parameter causes chunks to share characters."""
        result = chunk_text(
            "abcdefgh", chunk_size=4, overlap=2, metadata={"source_id": "doc1"}
        )

        assert [c.text for c in result] == ["abcd", "cdef", "efgh"]

    def test_chunk_id_format(self) -> None:
        """Chunk ID includes source_id and offsets."""
        result = chunk_text(
            "hello", chunk_size=10, overlap=0, metadata={"source_id": "doc1"}
        )

        assert result[0].chunk_id == "doc1:0:5"

    def test_chunk_id_uses_unknown_when_no_source_id(self) -> None:
        """Chunk ID defaults to 'unknown' when source_id not in metadata."""
        result = chunk_text("hello", chunk_size=10, overlap=0, metadata={})

        assert result[0].chunk_id == "unknown:0:5"

    def test_metadata_is_copied_to_each_chunk(self) -> None:
        """Each chunk gets a copy of metadata."""
        metadata = {"source_id": "doc1", "author": "test"}
        result = chunk_text("abcdefgh", chunk_size=4, overlap=0, metadata=metadata)

        assert all(c.metadata == metadata for c in result)
        assert (
            result[0].metadata is not result[1].metadata
        )  # Ensure copies, not same reference


class TestChunkTextValidation:
    def test_raises_on_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_text("text", chunk_size=0, overlap=0, metadata={})

    def test_raises_on_negative_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_text("text", chunk_size=-1, overlap=0, metadata={})

    def test_raises_on_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            chunk_text("text", chunk_size=5, overlap=-1, metadata={})

    def test_raises_when_overlap_equals_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < chunk_size"):
            chunk_text("text", chunk_size=5, overlap=5, metadata={})

    def test_raises_when_overlap_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < chunk_size"):
            chunk_text("text", chunk_size=5, overlap=10, metadata={})


class TestChunkDataclass:
    def test_chunk_is_frozen(self) -> None:
        """Chunk instances are immutable."""
        chunk = Chunk(
            chunk_id="id", text="text", offset_start=0, offset_end=4, metadata={}
        )

        with pytest.raises(AttributeError):
            chunk.text = "modified"  # type: ignore
