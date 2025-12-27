# llm-kit

**llm-kit** is a small, opinionated collection of reusable utilities for building **production-grade LLM and RAG systems**.

> **Extract the boring, repeated infrastructure that shows up across real LLM-backed services — and keep it minimal, explicit, and boring.**

---

## What llm-kit is NOT

- ❌ A framework
- ❌ A replacement for LangChain or LlamaIndex
- ❌ An agent orchestration engine
- ❌ A prompt engineering playground
- ❌ A research toolkit

Those tools are excellent at what they do. llm-kit focuses on **shared infrastructure**, not orchestration or experimentation.

---

## Quick Start

### Example 1: Embeddings + Vector Store

```python
from llm_kit.embeddings import LocalEmbeddingsClient
from llm_kit.vectorstores.qdrantvectorstore import QdrantVectorStore
from llm_kit.vectorstores.types import VectorItem

# Create embeddings client
embeddings = LocalEmbeddingsClient(model_name="all-MiniLM-L6-v2")

# Create vector store (in-memory for demo)
store = QdrantVectorStore(
    collection_name="docs",
    vector_size=384,
)

# Embed and store documents
texts = ["Python is great", "Rust is fast", "Go is simple"]
vectors = embeddings.embed(texts)

items = [
    VectorItem(id=f"doc-{i}", vector=v.vector, metadata={"text": t})
    for i, (t, v) in enumerate(zip(texts, vectors))
]
store.upsert(namespace="default", items=items)

# Query
query_vector = embeddings.embed(["programming languages"])[0].vector
results = store.query(namespace="default", vector=query_vector, top_k=2)

for r in results:
    print(f"{r.id}: {r.metadata['text']} (score: {r.score:.3f})")
```

### Example 2: Tool Calling

```python
from pydantic import BaseModel

from llm_kit.tools.tool import Tool, ToolCall
from llm_kit.tools.tool_engine import ToolEngine
from llm_kit.tools.tool_registry import ToolRegistry


# Define tool input schema
class WeatherInput(BaseModel):
    city: str


# Define tool handler
def get_weather(args: WeatherInput) -> str:
    return f"Weather in {args.city}: 22°C, sunny"


# Register tool
registry = ToolRegistry()
registry.register(
    Tool(
        name="get_weather",
        description="Get current weather for a city",
        input_schema=WeatherInput,
        handler=get_weather,
    )
)

# Execute tool call (e.g., from LLM response)
engine = ToolEngine(tool_registry=registry)
result = engine.call_tool(
    ToolCall(tool_name="get_weather", arguments={"city": "Tokyo"})
)
print(result)  # "Weather in Tokyo: 22°C, sunny"
```

---

## Design Philosophy

- **Thin abstractions only** — If an abstraction doesn't remove duplication across multiple apps, it doesn't belong here.
- **Constraints over flexibility** — Fewer supported options, chosen deliberately.
- **Explicit > clever** — No magic, no hidden behavior, no DSLs.
- **Infrastructure, not orchestration** — Business logic and agent flows stay in the application layer.
- **Easy to delete** — If llm-kit ever becomes more expensive than useful, it should be easy to remove.

---

## What's Included

### Embeddings

Standardized wrapper around embedding providers:

- `LocalEmbeddingsClient` — sentence-transformers
- `OpenAIEmbeddingsClient` — OpenAI API with retries

Features: batching, retries, timeouts, `metrics_hook` support.

### Chunking

Single, predictable chunking strategy:

```python
from llm_kit.chunking.chunking import chunk_text, Chunk

chunks = chunk_text(
    text="...",
    chunk_size=500,
    overlap=50,
    metadata={"source_id": "doc-1"},
)
```

No recursive chunking zoo. No document-type heuristics.

### Vector Stores

Narrow interface for vector databases:

- `PgVectorStore` — PostgreSQL with pgvector
- `QdrantVectorStore` — Qdrant
- `SQLiteVectorStore` — SQLite (zero-setup, perfect for dev/small datasets)

Exposes only: `upsert`, `query`, `delete`.

No query DSLs. No smart ranking logic.

### Tool Calling

Utilities to define and invoke tools safely:

- `Tool` — schema + handler
- `ToolRegistry` — register/lookup tools
- `ToolEngine` — validate and execute tool calls

Infrastructure glue, not agent logic.

### Prompt Registry

Simple prompt management:

- Prompts stored as YAML files
- Explicit versioning (name + version)
- Clear change history

```python
from llm_kit.prompts.prompts_library import PromptsLibrary

library = PromptsLibrary(directory="./prompts")
prompt = library.get(name="greeting", version="1.0")
```

No runtime mutation. No prompt DSLs.

### Observability

Optional metrics hook for all components:

```python
from llm_kit.observability.base import MetricsHook

class MyMetrics(MetricsHook):
    def record_latency(self, name: str, value_ms: float) -> None:
        # Send to your metrics backend
        pass

store = QdrantVectorStore(..., metrics_hook=MyMetrics())
```

All modules support `metrics_hook` for latency tracking.

---

## Non-Goals

These are explicitly **out of scope**:

- ❌ Agent frameworks
- ❌ Training or fine-tuning
- ❌ UI / dashboards
- ❌ Evaluation pipelines
- ❌ FastAPI integration in core

These belong elsewhere.

---

## Installation

```bash
poetry add git+https://github.com/sum7k/llm-kit.git
```

Or as a local path dependency during development.

---

## Who This Is For

llm-kit is useful if you:

- Build multiple LLM-backed services
- Care about engineering discipline
- Want consistency without over-engineering
- Prefer small, readable code over large frameworks

If you're looking for a feature-rich AI framework — this is not it.

---

## License

Unlicense license. See [LICENSE](LICENSE) for details.

---

> **llm-kit exists to make LLM systems boring — so the interesting problems stay in your application.**
