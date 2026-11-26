"""Offline stand-ins for the embedding model and reranker.

Used when the real bge models can't be downloaded (for example in a sandbox
with no access to huggingface.co). Scores produced with these are NOT the
scores of the real retriever: the vector side becomes a bag-of-words match
and the reranker a word-overlap count. Keyword search is unaffected.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by did does for from has have how in is it its of on or "
    "the to was were what which who why with".split()
)


class HashingEmbedder:
    """Deterministic hashed bag of words, 384 dims to fit the vector(384) column."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _WORD.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OverlapReranker:
    """Share of the query's non-stopword terms that appear in the passage."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        q = {t for t in _WORD.findall(query.lower()) if t not in _STOP}
        if not q:
            return [0.0] * len(passages)
        return [len(q & set(_WORD.findall(p.lower()))) / len(q) for p in passages]
