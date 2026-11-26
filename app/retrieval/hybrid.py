"""Hybrid retrieval: vector + keyword, fused with RRF, then cross-encoder reranked."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .embeddings import Embedder
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .models import RetrievedChunk
from .rerank import Reranker
from .store import PgSearchStore, SearchFilters

log = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    # GUESSES, tune once there is an eval set of diligence questions.
    vector_k: int = 50  # candidates from pgvector
    keyword_k: int = 50  # candidates from full-text search
    rrf_k: int = DEFAULT_RRF_K
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    rerank_top_n: int = 30  # how many fused candidates the cross-encoder sees
    top_k: int = 8  # chunks returned to generation
    min_rerank_score: float | None = None  # drop chunks below this, if set


class HybridRetriever:
    def __init__(
        self,
        store: PgSearchStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.config
        top_k = top_k or cfg.top_k
        query = query.strip()
        if not query:
            return []

        vector_hits = self.store.vector_search(
            self.embedder.embed_query(query), cfg.vector_k, filters
        )
        keyword_hits = self.store.keyword_search(query, cfg.keyword_k, filters)
        fused = reciprocal_rank_fusion(
            {"vector": vector_hits, "keyword": keyword_hits},
            k=cfg.rrf_k,
            weights={"vector": cfg.vector_weight, "keyword": cfg.keyword_weight},
        )
        log.debug(
            "hybrid query=%r vector=%d keyword=%d fused=%d",
            query, len(vector_hits), len(keyword_hits), len(fused),
        )

        if self.reranker is None:
            return fused[:top_k]

        pool = fused[: cfg.rerank_top_n]
        scores = self.reranker.score(query, [c.text for c in pool])
        for chunk, s in zip(pool, scores, strict=True):
            chunk.rerank_score = s
        pool.sort(key=lambda c: (-c.rerank_score, -c.rrf_score, c.chunk_id))
        if cfg.min_rerank_score is not None:
            pool = [c for c in pool if c.rerank_score >= cfg.min_rerank_score]
        return pool[:top_k]
