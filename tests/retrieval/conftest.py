"""Postgres fixtures for retrieval tests.

Set TEST_DATABASE_URL to a database with pgvector available. Tests that need
it are skipped otherwise. Each test runs in a throwaway schema.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

MIGRATIONS = sorted((REPO / "migrations").glob("*.sql"))  # 001 (ingestion) + 002 (retrieval)

DIM = 384


class HashingEmbedder:
    """Deterministic bag-of-words embedder: shared words -> similar vectors."""

    dim = DIM

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_query(self, text):
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def pg_conn():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    psycopg = pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    try:
        conn = psycopg.connect(url)
    except psycopg.OperationalError as e:
        pytest.skip(f"cannot connect to test database: {e}")
    schema = f"test_retrieval_{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}, public")
        for path in MIGRATIONS:
            cur.execute(path.read_text())
    conn.commit()
    try:
        yield conn
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.commit()
        conn.close()


CORPUS = [
    # (document_id, doc_type, company, section, content)
    ("doc_acme10k", "filing", "Acme", "Risk Factors",
     "Our three largest customers accounted for 62% of revenue. Loss of any of them "
     "would materially harm results. Customer concentration is a significant risk."),
    ("doc_acme10k", "filing", "Acme", "Liquidity and Capital Resources",
     "We had $45 million of cash and a $100 million undrawn revolving credit facility "
     "maturing in 2028. The facility contains a maximum leverage covenant of 3.5x."),
    ("doc_acme10k", "filing", "Acme", "Legal Proceedings",
     "A class action alleging patent infringement was filed against the company in Delaware."),
    ("doc_acmedeck", "investor_presentation", "Acme", "Growth Strategy",
     "We are expanding into the European market and launching a self-serve tier; "
     "management expects annual recurring revenue growth above 40%."),
    ("doc_betarpt", "market_report", "Beta", "Market Overview",
     "The industrial IoT market is projected to grow at a 14% CAGR through 2030, "
     "driven by predictive maintenance adoption."),
    ("doc_betarpt", "market_report", "Beta", "Competition",
     "Beta faces pricing pressure from low-cost competitors; customer churn rose to 9%."),
]


@pytest.fixture
def seeded(pg_conn, embedder):
    from app.retrieval.indexer import embed_pending_chunks

    docs = {}
    rows = []
    for i, (doc, dtype, company, section, content) in enumerate(CORPUS):
        docs[doc] = (dtype, company)
        rows.append((f"{doc}:c{i:04d}", doc, i, content, section,
                     {"doc_type": dtype, "company": company, "title": doc}))
    from psycopg.types.json import Jsonb

    with pg_conn.cursor() as cur:
        for doc, (dtype, company) in docs.items():
            cur.execute(
                "INSERT INTO documents (id, filename, title, doc_type, company, sha256, "
                "page_count, mime_type) VALUES (%s, %s, %s, %s, %s, %s, 1, 'application/pdf')",
                [doc, doc + ".pdf", doc, dtype, company, doc],
            )
        cur.executemany(
            "INSERT INTO chunks (id, document_id, chunk_index, content, section, metadata, "
            "kind, page_start, page_end) VALUES (%s, %s, %s, %s, %s, %s, 'text', 1, 1)",
            [(r[0], r[1], r[2], r[3], r[4], Jsonb(r[5])) for r in rows],
        )
    pg_conn.commit()
    n = embed_pending_chunks(pg_conn, embedder, model_name="hashing-test")
    assert n == len(CORPUS)
    return pg_conn
