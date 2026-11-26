# Technical notes

Details behind the [README](../README.md): API, ingestion, chunk schema, search modes and known gaps.

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

## Generation

`app/generation` turns retrieved chunks into a risk assessment, growth
opportunities and an executive summary. Every claim cites chunk IDs inline,
and claims without valid sources are dropped. It runs with **no API key**: the
default `demo` provider quotes the most relevant source sentences. Set
`GEMINI_API_KEY` (or `LLM_PROVIDER`) to get written analysis from a real model.
See `app/generation/README.md`.

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

## Upload-and-ask screen

A single page at `http://localhost:8000/` to upload filings, ask questions and generate the three reports.
Every claim shows numbered citations (document · page · section); click one to read the cited passage.
Claims the citation checker dropped are listed under each result.

No API key or paid service is needed.

```bash
pip install -e ".[dev]"
uvicorn app.main:app --reload
# open http://localhost:8000
```

The header shows which model answers:

- **demo (offline)**, the default: no AI model. It quotes the best-matching sentence from each retrieved
  passage, so upload, search, citations and the citation checker all work with nothing else installed.
- **local model via [Ollama](https://ollama.com)**: picked automatically when Ollama is running. Free, and
  documents never leave your machine. One-time setup: install Ollama, then `ollama pull qwen2.5:7b`
  (about 4.7 GB; `OLLAMA_MODEL` picks another model).
- Optional cloud models: a `GEMINI_API_KEY` switches to Gemini, or set `LLM_PROVIDER=anthropic`.
  Keys can go in a `.env` file, which is read at startup and never committed.

Search mode is shown in the header:

- **keyword (BM25)**, the default with no `DATABASE_URL`: in-memory, no extra installs, documents are lost on restart.
- **hybrid (vector + full-text + rerank)**: set `DATABASE_URL` to Postgres with pgvector, run
  `python -m app.storage.migrate`, and `pip install -e ".[retrieval]"`. New chunks are embedded before each search.
  The first run downloads the bge models. If anything is missing it falls back to keyword search and logs why.
  Force a mode with `DDC_RETRIEVAL=hybrid|keyword`; `DDC_RERANK=0` skips the cross-encoder.

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/ask` | `{question, document_ids?}` → cited answer + passages searched |
| `POST` | `/api/reports` | `{kind: risk_assessment \| growth_opportunities \| executive_summary, document_ids?, company?}` |
| `GET` | `/api/chunks/{chunk_id}` | One passage, for the citation viewer |
| `GET` | `/api/status` | Search mode and whether the LLM key is set |

Guesses to revisit: the canned search queries per report and pool size (`app/copilot/retrieve.py`), 8 passages
per question, up to 5 claims per answer, and `qwen2.5:7b` as the local model.
