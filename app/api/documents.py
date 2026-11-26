from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.ingestion import IngestError, ingest_bytes
from app.models import Chunk, DocType, Document
from app.storage import ChunkStore, get_store

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestResponse(BaseModel):
    document: Document
    chunk_count: int
    table_count: int
    sections: list[str]
    warnings: list[str]


@router.post("", status_code=status.HTTP_201_CREATED, response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: DocType = Form(DocType.other),
    company: str | None = Form(None),
    store: ChunkStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
    try:
        document, chunks = ingest_bytes(data, file.filename or "upload", doc_type, company or None, settings)
    except IngestError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.save(document, chunks)

    warnings = []
    if document.metadata.get("needs_ocr"):
        warnings.append("Little or no text layer found; this looks scanned and needs OCR (not supported yet).")
    sections = list(dict.fromkeys(c.section for c in chunks if c.section))
    return IngestResponse(
        document=document,
        chunk_count=len(chunks),
        table_count=sum(c.kind == "table" for c in chunks),
        sections=sections,
        warnings=warnings,
    )


@router.get("", response_model=list[Document])
def list_documents(store: ChunkStore = Depends(get_store)) -> list[Document]:
    return store.list_documents()


@router.get("/{document_id}", response_model=Document)
def get_document(document_id: str, store: ChunkStore = Depends(get_store)) -> Document:
    doc = store.get_document(document_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.get("/{document_id}/chunks", response_model=list[Chunk])
def get_chunks(
    document_id: str,
    page: int | None = Query(None, ge=1, description="Only chunks covering this page"),
    section: str | None = Query(None, description="Case-insensitive match on any heading in the section path"),
    store: ChunkStore = Depends(get_store),
) -> list[Chunk]:
    if not store.get_document(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return store.get_chunks(document_id, page=page, section=section)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, store: ChunkStore = Depends(get_store)) -> None:
    if not store.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
