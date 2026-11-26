# 10-K retrieval eval set (v1.1)

A small, hand-written evaluation set for checking whether the copilot's retriever
(hybrid search + reranking) brings back the right passages from SEC 10-K filings.

| | |
|---|---|
| Filings | Apple FY2026 (FYE 2026-09-26), Microsoft FY2026 (FYE 2026-06-30), NVIDIA FY2026 (FYE 2026-01-25), Tesla FY2025 (FYE 2025-12-31) |
| Questions | 55 (13 Apple, 13 Microsoft, 14 NVIDIA, 12 Tesla, 3 cross-company) |
| Question types | lookup 18, table_lookup 15, risk 6, calculation 4, explanation 4, unanswerable 3, comparison 3, false_premise 2 |
| Status | 47 verified, 5 derived, 3 guess |

## Files

- `10k_v1/corpus.json`: the filings, with EDGAR accession numbers and URLs.
- `10k_v1/questions.jsonl`: one question per line (schema below).
- `fetch_corpus.py`: downloads the filings from EDGAR into `10k_v1/docs/*.txt` (gitignored) and checks every evidence span against them.
- `score_retrieval.py`: scores a retriever's ranked results against the evidence spans.
- `score_answers.py`: checks generated answers for the right number (unit-aware, within tolerance) and for abstaining on unanswerable questions.
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
- **guess**: not confirmed against the filing. These are the three unanswerable questions: Apple Vision Pro units (`aapl-13`), Azure revenue in dollars (`msft-07`) and Blackwell units shipped (`nvda-14`). Each assumes the figure isn't disclosed. A targeted keyword search of each filing found nothing, but no filing was read in full.

Two question types test whether the system holds back. `unanswerable` questions have no evidence, and the right answer says the figure isn't disclosed. `false_premise` questions, such as how much NVIDIA invested in OpenAI, do have evidence, and the right answer corrects the premise.

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

# 4. Score answers (add an "answer" string to each results line)
python eval/score_answers.py results.jsonl

# Tests
python -m unittest discover -s eval/tests
```

Metrics: `evidence_recall@k` (share of spans covered in the top k), `full_hit@k` (all spans covered),
`doc_hit@k` (a chunk from a correct filing in the top k) and `mrr`. A span counts as covered
when it is a substring of a chunk after normalization, or when at least 90% of its tokens appear in
the chunk (`--overlap`). Unanswerable questions are skipped by the retrieval metrics. Use them to check
that the generator abstains.

## Known limits and next steps

- The set is small, so the numbers are noisy. Use it as a regression check and add questions over time.
- A span split across two chunks only counts if one chunk holds 90% or more of its tokens. Table-heavy questions are the most sensitive to how documents are chunked.
- `score_answers.py` only checks numbers and abstention. It doesn't grade the wording or check citations. An LLM judge or the generation module's citation validator could cover those.
- Possible additions: a bank (JPMorgan's 10-K points to an annual report exhibit, so ingestion has to follow exhibits), questions that span multiple years of one company's filings, and more false-premise questions.
