"""Reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009)."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Candidate, RetrievedChunk

DEFAULT_RRF_K = 60  # constant from the original paper; works well without tuning


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[Candidate]],
    k: int = DEFAULT_RRF_K,
    weights: dict[str, float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse several ranked candidate lists into one list ordered by RRF score.

    score(d) = sum over lists L of  w_L / (k + rank_L(d))

    `ranked_lists` maps a retriever name ("vector", "keyword") to its candidates,
    best first. Ranks are 1-based. Ties are broken by chunk_id for determinism.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    weights = weights or {}

    fused: dict[str, RetrievedChunk] = {}
    for name, candidates in ranked_lists.items():
        w = weights.get(name, 1.0)
        seen: set[str] = set()
        rank = 0
        for cand in candidates:
            if cand.chunk_id in seen:  # ignore duplicates within one list
                continue
            seen.add(cand.chunk_id)
            rank += 1
            item = fused.get(cand.chunk_id)
            if item is None:
                item = RetrievedChunk(
                    chunk_id=cand.chunk_id,
                    document_id=cand.document_id,
                    text=cand.text,
                    metadata=dict(cand.metadata),
                    rrf_score=0.0,
                )
                fused[cand.chunk_id] = item
            item.rrf_score += w / (k + rank)
            if name == "vector":
                item.vector_rank = rank
            elif name == "keyword":
                item.keyword_rank = rank
            else:
                item.extra[f"{name}_rank"] = rank

    return sorted(fused.values(), key=lambda c: (-c.rrf_score, c.chunk_id))
