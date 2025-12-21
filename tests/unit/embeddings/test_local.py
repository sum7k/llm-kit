from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest

from llm_kit.embeddings.base import Embedding
from llm_kit.embeddings.local import LocalEmbeddingsClient


@pytest.fixture
def mock_sentence_transformer() -> Generator[Mock, None, None]:
    """Fixture that patches SentenceTransformer and returns the mock model."""
    mock_model = Mock()
    with patch("llm_kit.embeddings.local.SentenceTransformer", return_value=mock_model):
        yield mock_model


def test_embed_returns_one_vector_per_input(mock_sentence_transformer: Mock) -> None:
    from llm_kit.embeddings.local import LocalEmbeddingsClient

    mock_sentence_transformer.encode.return_value = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    )
    client = LocalEmbeddingsClient(model_name="fake-model")

    embeddings = client.embed(["a", "b", "c"])

    assert len(embeddings) == 3
    assert all(isinstance(e, Embedding) for e in embeddings)
    assert all(isinstance(e.vector, list) for e in embeddings)


def test_embed_respects_batch_size(mock_sentence_transformer: Mock) -> None:
    """Test that texts are batched according to batch_size parameter."""
    from llm_kit.embeddings.local import LocalEmbeddingsClient

    mock_sentence_transformer.encode.side_effect = [
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        np.array([[0.7, 0.8, 0.9]]),
    ]
    client = LocalEmbeddingsClient(model_name="fake-model", batch_size=2)

    embeddings = client.embed(["a", "b", "c", "d", "e"])

    assert len(embeddings) == 5
    calls = mock_sentence_transformer.encode.call_args_list
    assert [len(call.args[0]) for call in calls] == [2, 2, 1]


def test_embed_raises_on_timeout(mock_sentence_transformer: Mock) -> None:
    mock_sentence_transformer.encode.side_effect = TimeoutError("Timeout occurred")
    client = LocalEmbeddingsClient(model_name="fake-model")
    texts = ["one sentence", "another sentence"]
    with pytest.raises(TimeoutError):
        client.embed(texts)


def test_embed_with_empty_input(mock_sentence_transformer: Mock) -> None:  # noqa: ARG001
    """Test that empty input list returns empty embeddings list."""
    client = LocalEmbeddingsClient(model_name="fake-model")
    embeddings = client.embed([])
    assert embeddings == []
    assert embeddings == []
