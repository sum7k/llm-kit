from unittest.mock import AsyncMock, Mock

import pytest
from openai import APITimeoutError

from llm_kit.embeddings.base import Embedding
from llm_kit.embeddings.openai import OpenAIEmbeddingsClient


def _mock_response(num_embeddings: int) -> Mock:
    """Create a mock response with the given number of embeddings."""
    return Mock(data=[Mock(embedding=[0.1, 0.2, 0.3]) for _ in range(num_embeddings)])


@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_input() -> None:
    client = OpenAIEmbeddingsClient(api_key="fake", model="fake-model")

    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = _mock_response(3)
    client._client = mock_client

    embeddings = await client.embed(["a", "b", "c"])

    assert len(embeddings) == 3
    assert all(isinstance(e, Embedding) for e in embeddings)
    assert all(isinstance(e.vector, list) for e in embeddings)


@pytest.mark.asyncio
async def test_embed_respects_batch_size() -> None:
    """Test that texts are batched according to batch_size parameter."""
    client = OpenAIEmbeddingsClient(api_key="fake", model="fake-model", batch_size=2)

    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = [
        _mock_response(2),
        _mock_response(2),
        _mock_response(1),
    ]
    client._client = mock_client

    embeddings = await client.embed(["a", "b", "c", "d", "e"])

    assert len(embeddings) == 5
    calls = mock_client.embeddings.create.call_args_list
    assert [len(call.kwargs["input"]) for call in calls] == [2, 2, 1]


@pytest.mark.asyncio
async def test_embed_raises_on_timeout() -> None:
    client = OpenAIEmbeddingsClient(api_key="fake", model="fake-model")

    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = APITimeoutError(request=Mock())  # type: ignore[arg-type]
    client._client = mock_client

    with pytest.raises(APITimeoutError):
        await client.embed(["test"])


@pytest.mark.asyncio
async def test_embed_with_empty_input() -> None:
    """Test that empty input list returns empty embeddings list."""
    client = OpenAIEmbeddingsClient(api_key="fake", model="fake-model")

    mock_client = AsyncMock()
    client._client = mock_client

    embeddings = await client.embed([])

    assert embeddings == []
    mock_client.embeddings.create.assert_not_called()
