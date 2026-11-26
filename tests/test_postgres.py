"""Runs only when TEST_DATABASE_URL points at a throwaway Postgres database (it drops tables)."""

import os

import pytest

from app.ingestion import ingest_bytes

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")]


@pytest.fixture
def pg_store():
    import psycopg

    from app.storage.migrate import migrate
    from app.storage.postgres import PostgresStore

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents, schema_migrations CASCADE")
    assert "001_documents_chunks.sql" in migrate(DSN)
    assert migrate(DSN) == []  # idempotent
    return PostgresStore(DSN)


def test_roundtrip(pg_store, sample_10k_pdf, settings):
    doc, chunks = ingest_bytes(sample_10k_pdf, "acme-10k.pdf", company="Acme", settings=settings)
    pg_store.save(doc, chunks)
    pg_store.save(doc, chunks)  # re-save replaces, no duplicate key error

    assert pg_store.get_document(doc.document_id).company == "Acme"
    assert [d.document_id for d in pg_store.list_documents()] == [doc.document_id]
    stored = pg_store.get_chunks(doc.document_id)
    assert [c.model_dump() for c in stored] == [c.model_dump() for c in chunks]
    risk = pg_store.get_chunks(doc.document_id, section="risk factors")
    assert risk and all(c.section == "Item 1A. Risk Factors" for c in risk)
    assert all(c.page_start <= 3 <= c.page_end for c in pg_store.get_chunks(doc.document_id, page=3))
    assert pg_store.get_chunk(chunks[0].chunk_id) == chunks[0]

    assert pg_store.delete(doc.document_id)
    assert pg_store.get_chunks(doc.document_id) == []
