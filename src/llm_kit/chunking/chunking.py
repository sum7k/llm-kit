from dataclasses import dataclass
from time import monotonic

from llm_kit.observability import names
from llm_kit.observability.base import MetricsHook, NoOpMetricsHook


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    offset_start: int
    offset_end: int
    metadata: dict


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    metadata: dict,
    metrics_hook: MetricsHook = NoOpMetricsHook(),
) -> list[Chunk]:
    start = monotonic()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    step = chunk_size - overlap
    text_len = len(text)

    for _, start in enumerate(range(0, text_len, step)):
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]

        chunk_id = f"{metadata.get('source_id', 'unknown')}:{start}:{end}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                offset_start=start,
                offset_end=end,
                metadata=dict(metadata),
            )
        )

        if end == text_len:
            break

    elapsed_ms = 1000 * (monotonic() - start)
    metrics_hook.record_latency(names.CHUNKING_DURATION, elapsed_ms)
    metrics_hook.increment(names.CHUNKING_CHUNKS_CREATED, len(chunks))
    return chunks
