"""Data types returned by the retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """A chunk returned by one first-stage retriever (vector or keyword)."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # raw score from that retriever (cosine similarity or ts_rank_cd)


@dataclass
class RetrievedChunk:
    """A chunk after fusion and reranking. This is what callers consume.

    Generation code should read: chunk_id, text, metadata, rerank_score.
    The other fields explain *why* a chunk was returned (useful for citations
    and debugging retrieval quality).
    """

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]
    rrf_score: float
    vector_rank: int | None = None  # 1-based rank in the vector list, None if absent
    keyword_rank: int | None = None  # 1-based rank in the keyword list, None if absent
    rerank_score: float | None = None  # cross-encoder score, None if reranking disabled
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Best available relevance score (rerank if present, else RRF)."""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score
