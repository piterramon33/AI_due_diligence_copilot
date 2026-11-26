"""Postgres store (psycopg 3). Apply the schema first with `python -m app.storage.migrate`."""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.models import Chunk, Document

_CHUNK_COLS = "id, document_id, chunk_index, content, kind, page_start, page_end, section, section_path, metadata"


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _connect(self) -> psycopg.Connection:
        # One short-lived connection per call keeps this simple; add psycopg_pool when traffic needs it.
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def save(self, document: Document, chunks: list[Chunk]) -> None:
        with self._connect() as conn, conn.transaction():
            conn.execute("DELETE FROM documents WHERE id = %s", (document.document_id,))
            conn.execute(
                """INSERT INTO documents (id, filename, title, doc_type, company, sha256, page_count,
                                          mime_type, ingested_at, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    document.document_id, document.filename, document.title, document.doc_type.value,
                    document.company, document.sha256, document.page_count, document.mime_type,
                    document.ingested_at, Jsonb(document.metadata),
                ),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO chunks ({_CHUNK_COLS}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.chunk_id, c.document_id, c.chunk_index, c.text, c.kind, c.page_start,
                            c.page_end, c.section, c.section_path, Jsonb(c.metadata),
                        )
                        for c in chunks
                    ],
                )

    def get_document(self, document_id: str) -> Document | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (document_id,)).fetchone()
        return _doc(row) if row else None

    def list_documents(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [_doc(r) for r in rows]

    def get_chunks(self, document_id: str, page: int | None = None, section: str | None = None) -> list[Chunk]:
        sql = f"SELECT {_CHUNK_COLS} FROM chunks WHERE document_id = %s"
        params: list = [document_id]
        if page is not None:
            sql += " AND page_start <= %s AND page_end >= %s"
            params += [page, page]
        if section is not None:
            sql += " AND EXISTS (SELECT 1 FROM unnest(section_path) s WHERE s ILIKE %s)"
            params.append(f"%{section}%")
        sql += " ORDER BY chunk_index"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_chunk(r) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_CHUNK_COLS} FROM chunks WHERE id = %s", (chunk_id,)).fetchone()
        return _chunk(row) if row else None

    def delete(self, document_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM documents WHERE id = %s", (document_id,)).rowcount > 0


def _doc(r: dict) -> Document:
    return Document(
        document_id=r["id"], filename=r["filename"], title=r["title"], doc_type=r["doc_type"],
        company=r["company"], sha256=r["sha256"], mime_type=r["mime_type"], page_count=r["page_count"],
        ingested_at=r["ingested_at"], metadata=r["metadata"] or {},
    )


def _chunk(r: dict) -> Chunk:
    return Chunk(
        chunk_id=r["id"], document_id=r["document_id"], chunk_index=r["chunk_index"], text=r["content"],
        kind=r["kind"], page_start=r["page_start"], page_end=r["page_end"], section=r["section"],
        section_path=list(r["section_path"] or []), metadata=r["metadata"] or {},
    )
