import logging
from typing import Any

from openai import OpenAI, OpenAIError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import EmbeddingsClient

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsClient(EmbeddingsClient):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        timeout: float = 10,
        batch_size: int = 100,
    ):
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self._batch_size = batch_size
        logger.info(
            "Initialized OpenAIEmbeddingsClient with model=%s, timeout=%s, batch_size=%s",
            model,
            timeout,
            batch_size,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("Empty input, returning empty list")
            return []

        logger.info("Embedding %d texts in batches of %d", len(texts), self._batch_size)
        embeddings: list[list[float]] = []

        for batch_num, start in enumerate(range(0, len(texts), self._batch_size), 1):
            end = start + self._batch_size
            batch_texts = texts[start:end]
            logger.debug(
                "Processing batch %d with %d texts", batch_num, len(batch_texts)
            )

            response = self._embed_batch(batch_texts)
            for data in response.data:
                embeddings.append(data.embedding)

        logger.info("Successfully embedded %d texts", len(embeddings))
        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(OpenAIError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> Any:
        return self._client.embeddings.create(
            model=self._model,
            input=batch,
        )
