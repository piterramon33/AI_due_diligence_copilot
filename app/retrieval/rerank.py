"""Second-stage reranking with a cross-encoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

# GUESS: bge-reranker-base is a good quality/speed trade-off for financial text.
# "cross-encoder/ms-marco-MiniLM-L-6-v2" is ~4x faster if latency matters more.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage (higher is better)."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, loaded lazily on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self._load().predict(pairs, batch_size=self.batch_size)
        return [float(s) for s in scores]
