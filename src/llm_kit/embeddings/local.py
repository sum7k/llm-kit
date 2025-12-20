# src/llm_kit/embeddings/local.py

from __future__ import annotations

import logging
from collections.abc import Iterable

from sentence_transformers import SentenceTransformer

from .base import EmbeddingsClient

logger = logging.getLogger(__name__)


class LocalEmbeddingsClient(EmbeddingsClient):
    """
    Local embedding client using sentence-transformers.

    This is a thin wrapper:
    - batching is internal
    - returns plain Python lists
    - no caching
    - no async
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize: bool = False,
    ) -> None:
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._normalize = normalize
        logger.info(
            "Initialized LocalEmbeddingsClient with model=%s, batch_size=%s, normalize=%s",
            model_name,
            batch_size,
            normalize,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("Empty input, returning empty list")
            return []

        logger.info("Embedding %d texts in batches of %d", len(texts), self._batch_size)
        embeddings: list[list[float]] = []

        for batch in _batch_iter(texts, self._batch_size):
            logger.debug("Processing batch with %d texts", len(batch))
            vectors = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )

            embeddings.extend(vectors.tolist())

        logger.info("Successfully embedded %d texts", len(embeddings))
        return embeddings


def _batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
