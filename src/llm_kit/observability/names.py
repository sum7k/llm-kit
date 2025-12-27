# src/llm_kit/observability/names.py

"""Standard metric names for llm-kit observability.

Use these constants instead of hardcoded strings for consistency
across the codebase.

Note: All duration metrics are in milliseconds by convention.
Units are handled by the metrics backend (e.g., converted to seconds in Prometheus).
"""

# ============================================================================
# LLM Metrics
# ============================================================================

# Duration
LLM_COMPLETION_DURATION = "llm_completion_duration"

# Counters
LLM_REQUESTS_TOTAL = "llm_requests_total"
LLM_ERRORS_TOTAL = "llm_errors_total"

# Counters (token usage - monotonic over time for cost/rate tracking)
LLM_TOKENS_PROMPT = "llm_tokens_prompt"
LLM_TOKENS_COMPLETION = "llm_tokens_completion"
LLM_TOKENS_TOTAL = "llm_tokens_total"


# ============================================================================
# Embeddings Metrics
# ============================================================================

# Duration
EMBEDDINGS_DURATION = "embeddings_duration"
# Legacy backend-specific names (prefer using EMBEDDINGS_DURATION with backend label)
EMBEDDINGS_OPENAI_DURATION = "openai_embeddings_duration"
EMBEDDINGS_LOCAL_DURATION = "local_embeddings_duration"

# Counters
EMBEDDINGS_REQUESTS_TOTAL = "embeddings_requests_total"
EMBEDDINGS_ERRORS_TOTAL = "embeddings_errors_total"

# Gauges
EMBEDDINGS_BATCH_SIZE = "embeddings_batch_size"


# ============================================================================
# Vector Store Metrics (PgVector)
# ============================================================================

# Duration
PGVECTOR_UPSERT_DURATION = "pgvector_upsert_duration"
PGVECTOR_QUERY_DURATION = "pgvector_query_duration"
PGVECTOR_DELETE_DURATION = "pgvector_delete_duration"

# Counters
PGVECTOR_OPERATIONS_TOTAL = "pgvector_operations_total"
PGVECTOR_ERRORS_TOTAL = "pgvector_errors_total"


# ============================================================================
# Vector Store Metrics (Qdrant)
# ============================================================================

# Duration
QDRANT_UPSERT_DURATION = "qdrant_upsert_duration"
QDRANT_QUERY_DURATION = "qdrant_query_duration"
QDRANT_DELETE_DURATION = "qdrant_delete_duration"

# Counters
QDRANT_OPERATIONS_TOTAL = "qdrant_operations_total"
QDRANT_ERRORS_TOTAL = "qdrant_errors_total"


# ============================================================================
# Tool Metrics
# ============================================================================

# Duration
TOOL_CALL_DURATION = "tool_call_duration"

# Counters
TOOL_CALLS_TOTAL = "tool_calls_total"
TOOL_ERRORS_TOTAL = "tool_errors_total"


# ============================================================================
# Chunking Metrics
# ============================================================================

# Duration
CHUNKING_DURATION = "chunking_duration"

# Counters (chunks accumulate over time)
CHUNKING_CHUNKS_CREATED = "chunking_chunks_created"
