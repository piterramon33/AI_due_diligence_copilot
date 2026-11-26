# Retrieval

Hybrid search over `chunks`: pgvector cosine similarity + Postgres full-text
search, merged with reciprocal rank fusion (k=60), then the top 30 are
reranked by a cross-encoder and the top 8 returned.

```
query ──► embed ──► pgvector top 50 ─┐
     └──────────► full-text top 50 ──┴─► RRF ─► top 30 ─► cross-encoder ─► top 8
```

## Pieces

| File | What it does |
|---|---|
| `migrations/002_hybrid_search.sql` | Adds `embedding vector(384)`, a generated `content_tsv`, HNSW + GIN indexes |
| `store.py` | `PgSearchStore.vector_search` / `keyword_search`, `SearchFilters` |
| `fusion.py` | `reciprocal_rank_fusion` |
| `rerank.py` | `CrossEncoderReranker` (default `BAAI/bge-reranker-base`) |
| `embeddings.py` | `SentenceTransformerEmbedder` (default `BAAI/bge-small-en-v1.5`) |
| `indexer.py` | `embed_pending_chunks` fills embeddings for chunks ingestion stored |
| `hybrid.py` | `HybridRetriever.retrieve(query, filters, top_k) -> list[RetrievedChunk]` |

`RetrievedChunk` fields: `chunk_id`, `document_id`, `text`, `metadata`
(chunk metadata plus `page_start`, `page_end`, `section`, `kind`,
`chunk_index`), `rrf_score`, `vector_rank`, `keyword_rank`, `rerank_score`,
and `score` (rerank score if present, else RRF).

## Dependencies

`psycopg[binary]>=3.1`, `pgvector>=0.2`, `numpy`, and for the real models
`sentence-transformers` (pulls in torch). Postgres needs the `vector`
extension, version 0.5 or newer for HNSW.

## Tests

```
TEST_DATABASE_URL=postgresql:///retrieval_test pytest tests/retrieval
```

Fusion tests run anywhere; the Postgres tests skip without `TEST_DATABASE_URL`.
They use a hashing embedder and a word-overlap reranker so they need no model
downloads.

## Guesses to revisit

- Embedding model and 384 dims. Changing it means editing the migration and re-embedding.
- Reranker model. `cross-encoder/ms-marco-MiniLM-L-6-v2` is faster if latency matters.
- Candidate sizes (50/50/30/8) and equal RRF weights. Tune with an eval set.
- Full-text ranking is `ts_rank_cd`, not true BM25. RRF only uses ranks, so this
  matters less, but ParadeDB `pg_search` would give real BM25.
- English text-search config only.
