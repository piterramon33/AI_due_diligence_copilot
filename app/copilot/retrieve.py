"""Retrieval for the UI: hybrid when available, BM25 keyword search otherwise."""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import Counter
from typing import Protocol, Sequence

from app.config import get_settings
from app.generation import ReportKind
from app.generation.types import ChunkLike
from app.storage import ChunkStore

log = logging.getLogger(__name__)

# GUESS: report-level queries. A report covers a whole company, so we run a
# few focused searches and pool the results rather than one vague query.
REPORT_QUERIES: dict[ReportKind, list[str]] = {
    ReportKind.RISK: [
        "risk factors",
        "liquidity indebtedness debt covenants cash flow",
        "customer concentration supplier dependence",
        "litigation legal proceedings regulatory investigation",
        "competition pricing pressure market share",
    ],
    ReportKind.GROWTH: [
        "growth strategy expansion new markets",
        "new products research and development pipeline",
        "revenue growth increased demand",
        "partnerships acquisitions investments",
        "margin improvement pricing efficiency",
    ],
    ReportKind.SUMMARY: [
        "business overview products and services",
        "total net sales revenue net income results of operations",
        "risk factors",
        "outlook strategy",
    ],
}
PER_QUERY_K = 6  # GUESS
REPORT_POOL_SIZE = 20  # GUESS: chunks sent to the LLM for one report
QUESTION_K = 8  # GUESS: chunks sent to the LLM for one question


class Retriever(Protocol):
    name: str

    def search(self, query: str, document_ids: Sequence[str] | None, k: int) -> list[ChunkLike]: ...

    def prepare(self) -> None:
        """Called before each search; indexes anything uploaded since the last call."""


# --- In-memory BM25 -----------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what which who will with how does do did our we their they".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class KeywordRetriever:
    """BM25 over every chunk in the store. Fine for a handful of filings.

    Rebuilds its index from the store when the set of documents changes, so it
    works with the in-memory store and with Postgres without the retrieval extras.
    """

    name = "keyword (BM25)"

    def __init__(self, store: ChunkStore, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self.k1, self.b = k1, b
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._chunks: list = []
        self._tfs: list[Counter] = []
        self._lens: list[int] = []
        self._df: Counter = Counter()

    def _refresh(self) -> None:
        docs = self.store.list_documents()
        key = tuple(sorted((d.document_id, d.ingested_at.isoformat()) for d in docs))
        if key == self._key:
            return
        chunks = [c for d in docs for c in self.store.get_chunks(d.document_id)]
        tfs = [Counter(tokenize(f"{c.section or ''}\n{c.text}")) for c in chunks]
        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        self._chunks, self._tfs, self._df = chunks, tfs, df
        self._lens = [sum(tf.values()) for tf in tfs]
        self._key = key

    def prepare(self) -> None:
        pass  # search() notices new or re-uploaded documents by itself

    def search(self, query: str, document_ids: Sequence[str] | None, k: int) -> list[ChunkLike]:
        with self._lock:
            self._refresh()
            chunks, tfs, lens, df = self._chunks, self._tfs, self._lens, self._df
        terms = set(tokenize(query))
        if not terms or not chunks:
            return []
        allowed = set(document_ids) if document_ids else None
        n = len(chunks)
        avg = sum(lens) / n or 1.0
        scored = []
        for i, (chunk, tf) in enumerate(zip(chunks, tfs)):
            if allowed is not None and chunk.document_id not in allowed:
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * lens[i] / avg))
            if score > 0:
                scored.append((score, chunk.chunk_id, chunk))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, _, c in scored[:k]]


# --- Hybrid (Postgres + pgvector + rerank) -----------------------------------


class HybridAdapter:
    """Wraps app.retrieval.HybridRetriever. Embeds chunks that have no embedding yet before searching."""

    name = "hybrid (vector + full-text + rerank)"

    def __init__(self, dsn: str) -> None:
        import psycopg

        from app.retrieval import (
            CrossEncoderReranker,
            HybridRetriever,
            PgSearchStore,
            SentenceTransformerEmbedder,
        )

        self._conn = psycopg.connect(dsn)
        self._embedder = SentenceTransformerEmbedder()
        rerank = os.environ.get("DDC_RERANK", "1") != "0"
        self._retriever = HybridRetriever(
            PgSearchStore(self._conn), self._embedder, CrossEncoderReranker() if rerank else None
        )
        self._lock = threading.Lock()

    def prepare(self) -> None:
        from app.retrieval import embed_pending_chunks

        with self._lock:
            n = embed_pending_chunks(self._conn, self._embedder, self._embedder.model_name)
        if n:
            log.info("embedded %d new chunks", n)

    def search(self, query: str, document_ids: Sequence[str] | None, k: int) -> list[ChunkLike]:
        from app.retrieval import SearchFilters

        filters = SearchFilters(document_ids=list(document_ids)) if document_ids else None
        with self._lock:
            return list(self._retriever.retrieve(query, filters, top_k=k))


_hybrid: Retriever | None = None
_hybrid_failed = False
_keyword: dict[int, KeywordRetriever] = {}
_lock = threading.Lock()


def get_retriever(store: ChunkStore) -> Retriever:
    """DDC_RETRIEVAL=hybrid|keyword. Default: hybrid with DATABASE_URL, else keyword.

    Hybrid falls back to keyword search (with a warning in the log) when the
    retrieval extras, pgvector, or migration 002 are missing.
    """
    global _hybrid, _hybrid_failed
    settings = get_settings()
    mode = os.environ.get("DDC_RETRIEVAL", "hybrid" if settings.database_url else "keyword")
    with _lock:
        if mode == "hybrid" and settings.database_url and not _hybrid_failed:
            if _hybrid is None:
                try:
                    _hybrid = HybridAdapter(settings.database_url)
                    log.info("retrieval: %s", _hybrid.name)
                except Exception as exc:
                    _hybrid_failed = True
                    log.warning("hybrid retrieval unavailable (%s); using keyword search", exc)
            if _hybrid is not None:
                return _hybrid
        if id(store) not in _keyword:
            _keyword[id(store)] = KeywordRetriever(store)
        return _keyword[id(store)]


def pool_for_report(
    retriever: Retriever, kind: ReportKind, document_ids: Sequence[str] | None
) -> list[ChunkLike]:
    """Round-robin merge of the per-query results, deduped, capped at REPORT_POOL_SIZE."""
    lists = [retriever.search(q, document_ids, PER_QUERY_K) for q in REPORT_QUERIES[kind]]
    pooled: dict[str, ChunkLike] = {}
    for rank in range(PER_QUERY_K):
        for hits in lists:
            if rank < len(hits):
                pooled.setdefault(hits[rank].chunk_id, hits[rank])
            if len(pooled) >= REPORT_POOL_SIZE:
                return list(pooled.values())
    return list(pooled.values())
