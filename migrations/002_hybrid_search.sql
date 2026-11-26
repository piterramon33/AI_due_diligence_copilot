-- 002: hybrid search columns and indexes on chunks.
-- Depends on 001_documents_chunks.sql (owned by ingestion).
--
-- GUESS: 384 dims matches BAAI/bge-small-en-v1.5 (app/retrieval/embeddings.py).
-- Changing the embedding model means changing this dimension and re-embedding.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS embedding vector(384),
    ADD COLUMN IF NOT EXISTS embedding_model text,
    ADD COLUMN IF NOT EXISTS embedded_at timestamptz;

-- Full-text vector, kept in sync by Postgres. The section heading is weighted
-- above body text so "Risk Factors" style headings pull their chunks up.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(section, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')), 'B')
    ) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);

-- HNSW for approximate cosine search (pgvector >= 0.5). Embeddings are
-- normalized, so cosine distance and inner product give the same order.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Filters used by retrieval.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_doc_type_idx ON chunks ((metadata ->> 'doc_type'));
CREATE INDEX IF NOT EXISTS chunks_metadata_company_idx ON chunks ((metadata ->> 'company'));
