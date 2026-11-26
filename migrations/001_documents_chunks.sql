-- 001: documents and chunks, owned by ingestion.
-- Search columns (embedding, content_tsv) are added by 002_ from the retrieval package.

CREATE TABLE IF NOT EXISTS documents (
    id           text PRIMARY KEY,            -- "doc_" + first 16 hex of sha256(file)
    filename     text NOT NULL,
    title        text NOT NULL,
    doc_type     text NOT NULL DEFAULT 'other'
                 CHECK (doc_type IN ('filing', 'financial_statement', 'investor_presentation', 'market_report', 'other')),
    company      text,
    sha256       text NOT NULL UNIQUE,
    page_count   integer NOT NULL,
    mime_type    text NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS chunks (
    id            text PRIMARY KEY,           -- "<document id>:c<4-digit index>"
    document_id   text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   integer NOT NULL,
    content       text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('text', 'table')),
    page_start    integer NOT NULL,
    page_end      integer NOT NULL,
    section       text,
    section_path  text[] NOT NULL DEFAULT '{}',
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,  -- source, title, doc_type, company, form_type
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS documents_company_idx ON documents (company);
