"""First-stage retrievers over Postgres: pgvector similarity and full-text search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .models import Candidate

_SELECT_COLS = """
    c.id, c.document_id, c.content,
    c.metadata || jsonb_build_object(
        'chunk_index', c.chunk_index,
        'kind', c.kind,
        'page_start', c.page_start,
        'page_end', c.page_end,
        'section', c.section
    ) AS metadata
"""


@dataclass
class SearchFilters:
    """Optional restrictions applied to both retrievers."""

    document_ids: Sequence[str] | None = None
    doc_types: Sequence[str] | None = None  # filing, financial_statement, ...
    companies: Sequence[str] | None = None
    kinds: Sequence[str] | None = None  # 'text' | 'table'
    metadata: dict[str, Any] = field(default_factory=dict)  # jsonb containment

    def to_sql(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.document_ids:
            clauses.append("c.document_id = ANY(%s)")
            params.append(list(self.document_ids))
        if self.doc_types:
            clauses.append("c.metadata ->> 'doc_type' = ANY(%s)")
            params.append(list(self.doc_types))
        if self.companies:
            clauses.append("c.metadata ->> 'company' = ANY(%s)")
            params.append(list(self.companies))
        if self.kinds:
            clauses.append("c.kind = ANY(%s)")
            params.append(list(self.kinds))
        if self.metadata:
            from psycopg.types.json import Jsonb

            clauses.append("c.metadata @> %s")
            params.append(Jsonb(self.metadata))
        sql = (" AND " + " AND ".join(clauses)) if clauses else ""
        return sql, params


def _uses_search_syntax(query: str) -> bool:
    q = f" {query} "
    return '"' in query or " -" in q or " OR " in q


def _row_to_candidate(row) -> Candidate:
    chunk_id, document_id, content, metadata, score = row
    return Candidate(
        chunk_id=chunk_id,
        document_id=document_id,
        text=content,
        metadata=metadata or {},
        score=float(score),
    )


class PgSearchStore:
    """Runs vector and keyword queries against the `chunks` table.

    `conn` is a psycopg 3 connection. The pgvector type adapter is registered on it.
    """

    def __init__(self, conn, text_search_config: str = "english", hnsw_ef_search: int = 100):
        from pgvector.psycopg import register_vector

        register_vector(conn)
        self.conn = conn
        self.text_search_config = text_search_config
        self.hnsw_ef_search = hnsw_ef_search

    def vector_search(
        self, query_embedding: Sequence[float], limit: int, filters: SearchFilters | None = None
    ) -> list[Candidate]:
        import numpy as np

        where, params = (filters or SearchFilters()).to_sql()
        vec = np.asarray(query_embedding, dtype=np.float32)
        sql = f"""
            SELECT {_SELECT_COLS}, 1 - (c.embedding <=> %s) AS score
            FROM chunks c
            WHERE c.embedding IS NOT NULL {where}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """
        with self.conn.transaction(), self.conn.cursor() as cur:
            # SET LOCAL only lasts for this transaction. Higher ef_search = better
            # recall when filters discard many HNSW candidates.
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)}")
            cur.execute(sql, [vec, *params, vec, limit])
            return [_row_to_candidate(r) for r in cur.fetchall()]

    def keyword_search(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[Candidate]:
        """Full-text search ranked by ts_rank_cd.

        NOTE: this is Postgres's cover-density ranking, not true BM25 (no IDF
        or length saturation). It is close enough for fusion, since RRF only
        uses ranks. ParadeDB's pg_search would give real BM25 if needed.

        Two query modes:
        - Plain questions ("what is the customer concentration risk") match
          ANY of the terms, so long natural-language questions still find
          chunks that mention only some of the words.
        - Queries using search syntax ("quoted phrase", -exclude, OR) go
          through websearch_to_tsquery and keep its strict semantics.
        """
        where, params = (filters or SearchFilters()).to_sql()
        cfg = self.text_search_config
        if _uses_search_syntax(query):
            tsq_sql = "websearch_to_tsquery(%s::regconfig, %s)"
        else:
            # Lexemes are already normalized by to_tsvector, so cast (not to_tsquery).
            tsq_sql = (
                "(SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery "
                "FROM unnest(to_tsvector(%s::regconfig, %s)))"
            )
        sql = f"""
            WITH q AS (SELECT {tsq_sql} AS tsq)
            SELECT {_SELECT_COLS}, ts_rank_cd(c.content_tsv, q.tsq, 32) AS score
            FROM chunks c, q
            WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq {where}
            ORDER BY score DESC, c.id
            LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, [cfg, query, *params, limit])
            return [_row_to_candidate(r) for r in cur.fetchall()]
