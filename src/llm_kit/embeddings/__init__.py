from .base import EmbeddingsClient
from .local import LocalEmbeddingsClient
from .openai import OpenAIEmbeddingsClient

__all__ = [
    "EmbeddingsClient",
    "LocalEmbeddingsClient",
    "OpenAIEmbeddingsClient",
]
