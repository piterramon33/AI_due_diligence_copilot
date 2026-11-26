"""ingest_bytes(): file bytes -> (Document, [Chunk]). Pure function; storage happens in the caller."""

from __future__ import annotations

import hashlib
from pathlib import PurePath

from app.config import Settings, get_settings
from app.ingestion.chunker import chunk_blocks
from app.ingestion.parsers import ParsedDocument, parse_html, parse_pdf, parse_text
from app.ingestion.sections import detect_form_type
from app.models import Chunk, DocType, Document

SUPPORTED = {
    ".pdf": ("application/pdf", parse_pdf),
    ".htm": ("text/html", parse_html),
    ".html": ("text/html", parse_html),
    ".txt": ("text/plain", parse_text),
    ".md": ("text/markdown", parse_text),
}


class IngestError(ValueError):
    """The file can't be ingested (unsupported type, corrupt, or no extractable text)."""


def document_id_for(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    return f"doc_{digest[:16]}", digest


def _sniff(filename: str, data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return ".pdf"
    ext = PurePath(filename).suffix.lower()
    if ext in SUPPORTED:
        return ext
    head = data[:1024].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in head:
        return ".html"
    raise IngestError(f"Unsupported file type for {filename!r}. Supported: {', '.join(sorted(SUPPORTED))}")


def ingest_bytes(
    data: bytes,
    filename: str,
    doc_type: DocType | str = DocType.other,
    company: str | None = None,
    settings: Settings | None = None,
) -> tuple[Document, list[Chunk]]:
    settings = settings or get_settings()
    if not data:
        raise IngestError("Empty file")
    doc_type = DocType(doc_type)
    ext = _sniff(filename, data)
    mime, parser = SUPPORTED[ext]
    try:
        parsed: ParsedDocument = parser(data)
    except Exception as exc:  # pdfminer/bs4 raise many unrelated types on corrupt input
        raise IngestError(f"Could not parse {filename!r}: {exc}") from exc

    document_id, digest = document_id_for(data)
    sample = " ".join(b.text for b in parsed.blocks[:200])
    metadata = dict(parsed.metadata)
    if form := detect_form_type(sample):
        metadata["form_type"] = form
        if doc_type is DocType.other:
            doc_type = DocType.filing  # guess: anything that says "FORM 10-K" etc. is a filing
    title = (parsed.title or PurePath(filename).stem).strip()

    document = Document(
        document_id=document_id,
        filename=filename,
        title=title,
        doc_type=doc_type,
        company=company,
        sha256=digest,
        mime_type=mime,
        page_count=parsed.page_count,
        metadata=metadata,
    )
    drafts = chunk_blocks(parsed.blocks, settings.chunk_max_words, settings.chunk_overlap_words)
    if not drafts and not metadata.get("needs_ocr"):
        raise IngestError(f"No extractable text in {filename!r}")

    shared = {"source": filename, "title": title, "doc_type": doc_type.value, "company": company}
    if "form_type" in metadata:
        shared["form_type"] = metadata["form_type"]
    chunks = [
        Chunk(
            chunk_id=Chunk.make_id(document_id, i),
            document_id=document_id,
            chunk_index=i,
            text=d.text,
            kind=d.kind,
            page_start=d.page_start,
            page_end=d.page_end,
            section=d.section_path[-1] if d.section_path else None,
            section_path=d.section_path,
            metadata=dict(shared),
        )
        for i, d in enumerate(drafts)
    ]
    document.metadata["chunk_count"] = len(chunks)
    return document, chunks
