# 10-K retrieval eval set (v1)

A small, hand-written evaluation set for checking whether the copilot's retriever
(hybrid search + reranking) brings back the right passages from SEC 10-K filings.

| | |
|---|---|
| Filings | Apple FY2026 (FYE 2026-09-26), NVIDIA FY2026 (FYE 2026-01-25), Tesla FY2025 (FYE 2025-12-31) |
| Questions | 38 (13 Apple, 12 NVIDIA, 11 Tesla, 2 cross-company) |
| Question types | lookup 12, table_lookup 11, risk 5, calculation 4, explanation 3, comparison 2, unanswerable 1 |
| Status | 33 verified, 4 derived, 1 guess |

## Files

- `10k_v1/corpus.json`: the filings, with EDGAR accession numbers and URLs.
- `10k_v1/questions.jsonl`: one question per line (schema below).
- `fetch_corpus.py`: downloads the filings from EDGAR into `10k_v1/docs/*.txt` (gitignored) and checks every evidence span against them.
- `score_retrieval.py`: scores a retriever's ranked results against the evidence spans.
- `tests/`: schema checks and scorer tests, including a fixture cut from real EDGAR table markup.

## Question schema

```json
{
  "id": "aapl-07",
  "doc_ids": ["AAPL-10K-FY2026"],
  "question": "How much did Apple spend on R&D in fiscal 2026 and what drove the increase?",
  "expected_answer": "$45,765 million, up 32% ...",
  "answer_value": 45765, "unit": "USD millions", "tolerance": 0.005,
  "evidence": ["Research and development $ 45,765 32 % $ 34,550", "R&D expense increased ..."],
  "section": "Item 7 - Operating Expenses",
  "question_type": "table_lookup",
  "difficulty": "medium",
  "status": "verified",
  "notes": "..."
}
```

- `evidence` holds verbatim spans from the filing. Whitespace is collapsed, and table cells are joined by single spaces, the way a browser's `innerText` or `fetch_corpus.py` renders them. A question is fully answerable only when every span is retrieved. The list is empty for `unanswerable` questions.
- `answer_value`, `unit` and `tolerance` are present when the answer is a single number. `tolerance` is relative.

## What is verified and what is a guess

- **verified**: every evidence span was found word for word in the filing's text from EDGAR on 2026-11-26, and the expected answer only restates those spans.
- **derived**: the evidence spans are verified, but the answer involves a hand calculation (a share, a growth rate, or free cash flow). Check the arithmetic if a number looks off.
- **guess**: not confirmed against the filing. Right now this is only `aapl-13`, the unanswerable Vision Pro units question. It assumes Apple doesn't disclose unit sales. A keyword search found nothing, but the filing was not read in full.

These fields are my own judgment on every question, so treat them as guesses: `difficulty`, `question_type`, and `section`. Section labels were placed by reading the surrounding text and are approximate. Notes on a question mark any specific doubt.

## Usage

```bash
# 1. Download the filings (SEC requires a User-Agent with contact info)
SEC_USER_AGENT="Your Name you@example.com" python eval/fetch_corpus.py

# 2. Run your retriever over each question and write results.jsonl:
#    {"id": "aapl-01", "retrieved": [{"doc_id": "AAPL-10K-FY2026", "text": "<chunk text>"}, ...]}
#    ranked best first, e.g. top 10 per question.

# 3. Score
python eval/score_retrieval.py results.jsonl            # summary
python eval/score_retrieval.py results.jsonl --json     # full per-question report

# Tests
python -m unittest discover -s eval/tests
```

## Running it against the real retriever

`run_retrieval.py` does steps 1-3 with the app's own code: it ingests the filings with
`app.ingestion` (production chunking) into Postgres, in a separate schema (`eval_10k_v1`),
embeds them with `app.retrieval.embed_pending_chunks`, asks every question through
`HybridRetriever`, and writes `eval/runs/<name>.jsonl` plus a scored `<name>.report.json`.

```bash
pip install -e ".[retrieval]"          # sentence-transformers is only needed for the bge models
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export SEC_USER_AGENT="Your Name you@example.com"

python eval/run_retrieval.py --fetch                         # full pipeline: bge-small + bge-reranker-base
python eval/run_retrieval.py --filter doc                    # search only the question's filings
python eval/run_retrieval.py --reranker none                 # hybrid without the reranker
python eval/run_retrieval.py --mode keyword --reranker none  # full-text search only, needs no models
python eval/run_retrieval.py --embedder hashing --reranker overlap   # offline stand-ins, plumbing only
```

`--fetch` saves the raw EDGAR HTML as `10k_v1/docs/<doc_id>.htm`, which keeps tables for the
ingestion parser. A `.txt` from `fetch_corpus.py` also works, but tables arrive flattened.
Re-runs reuse the ingested chunks and embeddings unless chunk settings
(`DDC_CHUNK_MAX_WORDS`, `DDC_CHUNK_OVERLAP_WORDS`) or the embedding model change; `--reset` starts over.

Besides the metrics, the report has `chunking_ceiling`: the share of evidence spans that appear in at
least one stored chunk. A span missing there is lost to parsing or chunking, not to search.
Runs with `--embedder hashing` or `--reranker overlap` are marked `standins` in the report and are
not scores of the real models.

Metrics: `evidence_recall@k` (share of spans covered in the top k), `full_hit@k` (all spans covered),
`doc_hit@k` (a chunk from a correct filing in the top k) and `mrr`. A span counts as covered
when it is a substring of a chunk after normalization, or when at least 90% of its tokens appear in
the chunk (`--overlap`). Unanswerable questions are skipped by the retrieval metrics. Use them to check
that the generator abstains.

## Known limits and next steps

- The set is small, so the numbers are noisy. Use it as a regression check and add questions over time.
- A span split across two chunks only counts if one chunk holds 90% or more of its tokens. Table-heavy questions are the most sensitive to how documents are chunked.
- Answer quality is not scored yet. `answer_value` and `tolerance` are there so a numeric answer check can be added.
- Possible additions: more filings (a bank, such as JPMorgan, whose 10-K points to an annual report exhibit), more unanswerable questions, and questions that need Item 1A risk factors and the financial statements together.
