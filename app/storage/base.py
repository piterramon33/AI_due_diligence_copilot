from __future__ import annotations

from typing import Protocol

from app.models import Chunk, Document


class ChunkStore(Protocol):
    def save(self, document: Document, chunks: list[Chunk]) -> None:
        """Insert or replace a document and all its chunks atomically."""

    def get_document(self, document_id: str) -> Document | None: ...

    def list_documents(self) -> list[Document]: ...

    def get_chunks(
        self, document_id: str, page: int | None = None, section: str | None = None
    ) -> list[Chunk]: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def delete(self, document_id: str) -> bool: ...
