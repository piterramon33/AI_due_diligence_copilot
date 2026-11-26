# AI Due Diligence Copilot

RAG-powered platform that analyzes company filings, financial statements, investor presentations and market
reports to speed up due diligence. Hybrid search and reranking retrieve the relevant passages, and generated
risk assessments, growth opportunities and executive summaries cite their sources.

This branch holds the **backend skeleton and document ingestion**. Retrieval (`app/retrieval/`) and cited
generation (`app/generation/`) build on the chunk model defined here.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,postgres]"
uvicorn app.main:app --reload          # http://localhost:8000/docs
pytest
```

Without `DATABASE_URL` documents are kept in memory (lost on restart). For Postgres:

```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/diligence
python -m app.storage.migrate          # applies migrations/*.sql once each
TEST_DATABASE_URL=$DATABASE_URL pytest # also runs the Postgres tests (they drop tables: use a scratch DB)
```

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/documents` | Multipart upload: `file`, optional `doc_type`, `company`. Parses, chunks and stores. Re-uploading the same file replaces it. |
| `GET` | `/documents` | List documents |
| `GET` | `/documents/{id}` | One document |
| `GET` | `/documents/{id}/chunks?page=&section=` | Chunks, optionally only those covering a page or under a heading |
| `DELETE` | `/documents/{id}` | Remove a document and its chunks |
| `GET` | `/health` | Liveness |

`doc_type` is one of `filing`, `financial_statement`, `investor_presentation`, `market_report`, `other`.
Files that say "FORM 10-K" (10-Q, 8-K, 20-F, S-1 ...) are tagged `filing` with `metadata.form_type`.

## Ingestion

```
bytes ─► parser (PDF | EDGAR HTML | text/Markdown) ─► ordered blocks (heading / text / table, each with a page)
      ─► section tracking (Part ▸ Item ▸ heading) ─► chunker ─► Chunk[] with citation metadata
```

- **PDF** (pdfplumber): lines grouped into paragraphs, ruled tables extracted separately (and removed from the
  prose), headings from `PART`/`ITEM` patterns or larger fonts, running headers/footers and page numbers dropped.
  Scanned PDFs are flagged `needs_ocr`.
- **EDGAR HTML**: page numbers recovered from CSS page breaks, hidden inline-XBRL dropped, `$`/`%` cells merged
  into their numbers, colspans expanded so financial tables line up.
- **Text / Markdown**: `#` headings and SEC items as sections, form feeds as page breaks.

Every chunk carries what a citation needs:

```python
Chunk(
    chunk_id="doc_ab12cd34ef567890:c0007",   # "<document_id>:c<index>", document_id = "doc_" + sha256[:16]
    document_id="doc_ab12cd34ef567890",
    chunk_index=7,
    text="...",                              # tables are Markdown
    kind="text" | "table",
    page_start=12, page_end=13,              # 1-based, exact
    section="Item 1A. Risk Factors",
    section_path=["Part I", "Item 1A. Risk Factors"],
    metadata={"source": "acme-10k.pdf", "title": "...", "doc_type": "filing", "company": "Acme", "form_type": "10-K"},
)
chunk.citation()  # "acme-10k.pdf, p. 12-13, Item 1A. Risk Factors"
```

Chunks never cross a section; prose packs whole paragraphs up to `DDC_CHUNK_MAX_WORDS` (350) with
`DDC_CHUNK_OVERLAP_WORDS` (50) overlap; each table is its own chunk, split by rows with the header repeated.

## Layout

```
app/
  main.py, config.py, models.py   # FastAPI app, settings, shared Document/Chunk models
  api/documents.py                # upload and read endpoints
  ingestion/                      # parsers/, sections.py, chunker.py, pipeline.py
  storage/                        # ChunkStore protocol, in-memory and Postgres stores, migrate runner
migrations/001_documents_chunks.sql
tests/                            # synthetic 10-K PDF/HTML built at test time
```

## Known gaps (first version)

- Tables without ruling lines in PDFs (common in decks and some financial statements) come out as prose.
- No OCR for scanned PDFs yet; they are flagged, not parsed.
- A 10-K table of contents becomes a table chunk with no section.
- Heading thresholds and chunk sizes are guesses, not yet tuned on real filings.
- No `.docx`, `.xlsx` or `.pptx` parsers yet.
