from __future__ import annotations

import threading

from app.models import Chunk, Document


def filter_chunks(chunks: list[Chunk], page: int | None, section: str | None) -> list[Chunk]:
    if page is not None:
        chunks = [c for c in chunks if c.page_start <= page <= c.page_end]
    if section is not None:
        s = section.lower()
        chunks = [c for c in chunks if any(s in p.lower() for p in c.section_path)]
    return chunks


class InMemoryStore:
    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._lock = threading.Lock()

    def save(self, document: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            self._docs[document.document_id] = document
            self._chunks[document.document_id] = list(chunks)

    def get_document(self, document_id: str) -> Document | None:
        return self._docs.get(document_id)

    def list_documents(self) -> list[Document]:
        return sorted(self._docs.values(), key=lambda d: d.ingested_at, reverse=True)

    def get_chunks(self, document_id: str, page: int | None = None, section: str | None = None) -> list[Chunk]:
        return filter_chunks(self._chunks.get(document_id, []), page, section)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        document_id = chunk_id.split(":", 1)[0]
        return next((c for c in self._chunks.get(document_id, []) if c.chunk_id == chunk_id), None)

    def delete(self, document_id: str) -> bool:
        with self._lock:
            self._chunks.pop(document_id, None)
            return self._docs.pop(document_id, None) is not None
