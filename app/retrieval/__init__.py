"""Hybrid retrieval: pgvector + Postgres full-text, RRF fusion, cross-encoder rerank.

Typical use:

    import psycopg
    from app.retrieval import (
        HybridRetriever, PgSearchStore, SentenceTransformerEmbedder, CrossEncoderReranker,
    )

    conn = psycopg.connect(os.environ["DATABASE_URL"])
    retriever = HybridRetriever(
        PgSearchStore(conn), SentenceTransformerEmbedder(), CrossEncoderReranker()
    )
    chunks = retriever.retrieve("customer concentration risk", SearchFilters(companies=["Acme"]))
"""

from .embeddings import Embedder, SentenceTransformerEmbedder
from .fusion import reciprocal_rank_fusion
from .hybrid import HybridRetriever, RetrievalConfig
from .indexer import embed_pending_chunks
from .models import Candidate, RetrievedChunk
from .rerank import CrossEncoderReranker, Reranker
from .store import PgSearchStore, SearchFilters

__all__ = [
    "Candidate",
    "CrossEncoderReranker",
    "Embedder",
    "HybridRetriever",
    "PgSearchStore",
    "Reranker",
    "RetrievalConfig",
    "RetrievedChunk",
    "SearchFilters",
    "SentenceTransformerEmbedder",
    "embed_pending_chunks",
    "reciprocal_rank_fusion",
]
