"""Query/passage embedding. Pluggable so tests and ingestion can swap models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

# GUESS: bge-small is a cheap, strong English default (384 dims). Must match
# whatever the ingestion pipeline uses to embed chunks.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIM = 384
# bge models want this prefix on queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers (loaded lazily)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        query_prefix: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        if query_prefix is None:
            query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.query_prefix = query_prefix
        self.device = device
        self._model = None
        self.dim = DEFAULT_EMBEDDING_DIM if model_name == DEFAULT_EMBEDDING_MODEL else -1

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_query(self, text: str) -> list[float]:
        vec = self._load().encode(self.query_prefix + text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._load().encode(list(texts), normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]
