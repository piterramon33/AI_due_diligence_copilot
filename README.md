# AI Due Diligence Copilot

**Ask questions of a 10-K and get answers where every claim cites the exact page and section it came from.**

A retrieval-augmented generation (RAG) app for due diligence. It parses company filings, financial
statements, investor decks and market reports, finds the relevant passages with hybrid search and reranking, and
writes risk assessments, growth opportunities and executive summaries. A citation checker drops any claim that its
sources don't support.

It runs with **no API key and no paid service**. Open it and Apple's FY2026 Form 10-K, taken unchanged from SEC
EDGAR, is already loaded.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/piterramon33/AI_due_diligence_copilot)

![Asking about Apple's net sales: the answer is a row of the income statement, cited to page 28 of the 10-K](docs/img/ask.png)

| Click a citation to read the source passage | Reports built from many searches |
| --- | --- |
| ![Citation drawer showing the net sales table from Item 7](docs/img/citation.png) | ![Risk assessment with severity ratings and citations to Item 1A](docs/img/report.png) |

## What it does

- **Reads real filings.** It handles EDGAR HTML (inline XBRL removed, page numbers recovered from page breaks,
  financial tables rebuilt), PDFs (paragraphs, ruled tables, headings, running headers removed) and text or
  Markdown. Chunks never cross a section, and each one keeps its page range and Part › Item › heading path.
- **Hybrid retrieval.** pgvector embeddings (`bge-small-en-v1.5`) and Postgres full-text search are fused with
  Reciprocal Rank Fusion, then reranked with a cross-encoder (`bge-reranker-base`). With no database it falls
  back to in-memory BM25, which is what the hosted demo uses.
- **Cited generation.** Every claim must cite chunk IDs inline. The checker rejects claims with no citation,
  unknown IDs, or **numbers that don't appear in the cited passage**. The model gets one repair pass, and anything
  that still fails is shown as "dropped by the citation check".
- **Three diligence reports** besides free-form Q&A: risk assessment (rated by severity), growth opportunities
  and executive summary. Each one pools several targeted searches.
- **An evaluation set** of 55 questions over four real 10-Ks (Apple, Microsoft, NVIDIA, Tesla). It includes
  unanswerable and false-premise questions, with scorers for retrieval recall and answer accuracy (`eval/`).

## How it works

```mermaid
flowchart LR
  subgraph Ingest
    F[PDF / EDGAR HTML / text] --> P[Parser<br/>blocks + pages]
    P --> S[Section tracker<br/>Part › Item › heading]
    S --> C[Chunker<br/>tables kept whole]
  end
  C --> DB[(Postgres + pgvector<br/>or in-memory)]
  Q[Question or report] --> R
  subgraph Retrieve
    R[Vector search<br/>bge-small] --> RRF[RRF fusion]
    K[Full-text / BM25] --> RRF
    RRF --> RR[Cross-encoder rerank]
  end
  DB --> R & K
  RR --> G[LLM drafts claims<br/>with chunk citations]
  G --> V{Citation check<br/>ids, coverage, numbers}
  V -- pass --> A[Answer with<br/>page + section cites]
  V -- fail --> X[Repair once, then drop]
```

## Choose the model, all free by default

| Mode | Setup | What answers |
| --- | --- | --- |
| **Demo (default)** | nothing | No AI model. It quotes the best-matching source sentences and financial-table rows, so search, citations and the checker all work offline. |
| **Local model** | [Ollama](https://ollama.com) running, plus `ollama pull qwen2.5:7b` | Picked automatically when Ollama is running. Written answers, and documents never leave your machine. |
| Gemini / Claude (optional) | `GEMINI_API_KEY`, or `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY` | Cloud models, used only if you opt in. |

## Run it

**Docker** (same image as the hosted demo):

```bash
docker build -t diligence-copilot .
docker run -p 7860:7860 diligence-copilot      # http://localhost:7860
```

**Python 3.11+**:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload                  # http://localhost:8000, API docs at /docs
pytest
```

**Full hybrid search** (Postgres with pgvector, local embedding and reranking models):

```bash
pip install -e ".[retrieval]"
export DATABASE_URL=postgresql://user:pass@localhost:5432/diligence
python -m app.storage.migrate
uvicorn app.main:app
```

## Deploy a free public demo

The demo needs no database, model download or key, and the image runs in about 70 MB of RAM.

- **Render:** click the button above, or create a *Blueprint* from this repo. `render.yaml` sets the free plan.
  Free instances sleep when idle, so the first visit after a while takes about a minute to wake up.
- **Hugging Face Spaces:** create a *Docker* Space and push this repo to it. Spaces read their settings from YAML
  at the top of the Space's README, so add `sdk: docker` and `app_port: 7860` there.

`DDC_PUBLIC_DEMO=1` (set in the Dockerfile) caps uploads at 10 MB, keeps only the 5 newest uploads, protects the
sample filing from deletion, and warns visitors that uploads are shared. Set `DDC_DEMO_PRELOAD=0` to start empty.

## Project layout

```
app/
  ingestion/    parsers (PDF, EDGAR HTML, text), section tracking, chunker
  storage/      in-memory and Postgres stores, SQL migrations runner
  retrieval/    embeddings, pgvector + full-text search, RRF fusion, cross-encoder rerank
  generation/   prompts, LLM providers (demo, Ollama, Gemini, Claude), citation validator, reports
  copilot/      Q&A, report search pools, BM25 fallback, offline demo answerer
  api/          REST endpoints;  static/index.html is the single-page UI
  demo.py       preloads the bundled Apple 10-K (demo_data/)
eval/           10-K question set, retrieval runner and scorers
migrations/     001 documents/chunks, 002 hybrid search columns
```

More detail (API reference, chunk schema, ingestion rules, known gaps) is in [docs/technical.md](docs/technical.md).

## Honest limits

- Demo mode quotes passages and doesn't reason. For written analysis, run a local Ollama model.
- Retrieval sizes, chunk sizes and model choices are reasonable defaults, not yet tuned on the eval set.
- There is no OCR for scanned PDFs, and no `.docx`, `.xlsx` or `.pptx` parsing yet.
- This is a portfolio project, not investment advice.

Sample data: Apple Inc. Form 10-K for the fiscal year ended September 26, 2026, from
[SEC EDGAR](https://www.sec.gov/Archives/edgar/data/320193/000032019326000026/aapl-20260926.htm) (public record).
