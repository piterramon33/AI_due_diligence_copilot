"""Fills chunks.embedding for chunks that ingestion stored without one."""

from __future__ import annotations

from .embeddings import Embedder


def embed_pending_chunks(conn, embedder: Embedder, model_name: str, batch_size: int = 64) -> int:
    """Embed every chunk whose embedding is NULL (or from another model).

    Returns the number of chunks embedded. Safe to re-run.
    """
    import numpy as np
    from pgvector.psycopg import register_vector

    register_vector(conn)
    total = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, coalesce(section || E'\n', '') || content
                FROM chunks
                WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM %s
                ORDER BY id
                LIMIT %s
                """,
                [model_name, batch_size],
            )
            rows = cur.fetchall()
            if not rows:
                return total
            vecs = embedder.embed_documents([r[1] for r in rows])
            cur.executemany(
                """
                UPDATE chunks
                SET embedding = %s, embedding_model = %s, embedded_at = now()
                WHERE id = %s
                """,
                [(np.asarray(v, dtype=np.float32), model_name, r[0]) for r, v in zip(rows, vecs)],
            )
        conn.commit()
        total += len(rows)
