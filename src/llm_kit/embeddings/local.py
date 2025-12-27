# src/llm_kit/embeddings/local.py

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from time import monotonic

from sentence_transformers import SentenceTransformer

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import Embedding, EmbeddingsClient

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
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ) -> None:
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._normalize = normalize
        self.metrics_hook = metrics_hook
        logger.info(
            "Initialized LocalEmbeddingsClient with model=%s, batch_size=%s, normalize=%s",
            model_name,
            batch_size,
            normalize,
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            logger.debug("Empty input, returning empty list")
            return []

        start = monotonic()
        logger.info("Embedding %d texts in batches of %d", len(texts), self._batch_size)
        embeddings: list[Embedding] = []

        for batch in _batch_iter(texts, self._batch_size):
            logger.debug("Processing batch with %d texts", len(batch))
            # Use to_thread to avoid blocking event loop with CPU-bound work
            vectors = await asyncio.to_thread(
                self._model.encode,
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )

            embeddings.extend(Embedding(vector=v.tolist()) for v in vectors)

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.EMBEDDINGS_LOCAL_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.EMBEDDINGS_REQUESTS_TOTAL, labels={"backend": "local"}
        )
        logger.info("Successfully embedded %d texts", len(embeddings))
        return embeddings


def _batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
