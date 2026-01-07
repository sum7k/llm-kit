import asyncio
import logging
from time import monotonic
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook

from .base import Embedding, EmbeddingsClient

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsClient(EmbeddingsClient):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float = 10,
        batch_size: int = 100,
        max_concurrent: int = 3,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ):
        """
        Initialize OpenAI embeddings client.

        Args:
            api_key: OpenAI API key. If None, falls back to OPENAI_API_KEY env var.
            model: Embedding model to use.
            timeout: Request timeout in seconds.
            batch_size: Number of texts to embed per batch.
            max_concurrent: Maximum concurrent API requests. Limits parallelism to avoid
                rate limiting. Set to 1 for sequential processing.
            metrics_hook: Hook for recording metrics.
        """
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics_hook = metrics_hook
        logger.info(
            "Initialized OpenAIEmbeddingsClient with model=%s, timeout=%s, batch_size=%s, max_concurrent=%s",
            model,
            timeout,
            batch_size,
            max_concurrent,
        )

    async def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            logger.debug("Empty input, returning empty list")
            return []

        start = monotonic()
        logger.info("Embedding %d texts in batches of %d", len(texts), self._batch_size)

        # Process all batches concurrently
        batches = []
        for batch_start in range(0, len(texts), self._batch_size):
            end = batch_start + self._batch_size
            batches.append(texts[batch_start:end])

        logger.debug(
            "Processing %d batches with max %d concurrent",
            len(batches),
            self._semaphore._value,
        )
        responses = await asyncio.gather(
            *[self._embed_batch_with_semaphore(batch) for batch in batches]
        )

        # Flatten results
        embeddings: list[Embedding] = []
        for response in responses:
            for data in response.data:
                embeddings.append(Embedding(vector=data.embedding))

        elapsed_ms = 1000 * (monotonic() - start)
        self.metrics_hook.record_latency(names.EMBEDDINGS_OPENAI_DURATION, elapsed_ms)
        self.metrics_hook.increment(
            names.EMBEDDINGS_REQUESTS_TOTAL, labels={"backend": "openai"}
        )
        logger.info("Successfully embedded %d texts", len(embeddings))
        return embeddings

    async def _embed_batch_with_semaphore(self, batch: list[str]) -> Any:
        """Embed a batch with semaphore to limit concurrent requests."""
        async with self._semaphore:
            return await self._embed_batch(batch)

    async def _embed_batch(self, batch: list[str]) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
            retry=retry_if_exception_type(OpenAIError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                return await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
