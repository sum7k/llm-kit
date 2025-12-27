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
        api_key: str,
        model: str = "text-embedding-ada-002",
        timeout: float = 10,
        batch_size: int = 100,
        metrics_hook: MetricsHook = NoOpMetricsHook(),
    ):
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._batch_size = batch_size
        self.metrics_hook = metrics_hook
        logger.info(
            "Initialized OpenAIEmbeddingsClient with model=%s, timeout=%s, batch_size=%s",
            model,
            timeout,
            batch_size,
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

        logger.debug("Processing %d batches concurrently", len(batches))
        responses = await asyncio.gather(
            *[self._embed_batch(batch) for batch in batches]
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
