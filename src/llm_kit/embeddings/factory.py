# src/llm_kit/embeddings/factory.py

from .base import EmbeddingsClient
from .config import EmbeddingsConfig
from .local import LocalEmbeddingsClient
from .openai import OpenAIEmbeddingsClient


def create_embeddings_client(
    config: EmbeddingsConfig,
) -> EmbeddingsClient:
    if config.provider == "openai":
        return OpenAIEmbeddingsClient(
            api_key=config.api_key or "",
            model=config.model,
            timeout=config.timeout,
            batch_size=config.batch_size,
        )

    if config.provider == "local":
        return LocalEmbeddingsClient(
            model_name=config.model,
            batch_size=config.batch_size,
        )

    raise ValueError(f"Unknown embeddings provider: {config.provider}")
