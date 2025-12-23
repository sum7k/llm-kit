CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE vector_items (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    metadata JSONB NOT NULL,

    PRIMARY KEY (namespace, id)
);

-- Required for similarity search
CREATE INDEX vector_items_embedding_idx
ON vector_items
USING ivfflat (embedding vector_cosine_ops);
