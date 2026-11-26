# Cited generation

Turns retrieved chunks into three reports: risk assessment, growth opportunities,
and executive summary. Every claim cites chunk IDs inline, like
`Top three customers are 46% of revenue [doc_ab12cd34ef567890:c0007].`
Claims that fail the source check are dropped from the report and kept in
`report.rejected` with the reasons.

```python
from app.generation import ReportGenerator, get_llm

gen = ReportGenerator(get_llm())               # keyless demo unless a key is set
report = gen.risk_assessment("Acme Corp", chunks)  # chunks from retrieval
print(report.to_markdown())
```

## Providers

| `LLM_PROVIDER` | Needs | What you get |
|---|---|---|
| `demo` (default with no key) | nothing | Extractive: quotes the best-matching source sentences with their chunk IDs. No API, no cost, runs offline. |
| `gemini` (default when `GEMINI_API_KEY` is set) | `pip install .[gemini]` and a Google AI Studio key | Written analysis from `gemini-3.8-flash` |
| `anthropic` | `pip install .[anthropic]` and `ANTHROPIC_API_KEY` | Written analysis from Claude |

The demo provider goes through the same citation checks as the real models,
so the whole pipeline can be shown without any account or key.

## What gets rejected

- no inline `[chunk_id]` citation, or a sentence without one
- a cited ID that was not in the chunks given to the model
- a `citations` list that names IDs not cited inline
- any number not written in one of the chunks that claim cites
- a rating other than low / medium / high

Failed claims get one repair pass (`max_repairs=1`), then are dropped.

## Guesses to confirm

- **Gemini model:** `gemini-3.8-flash`, the strongest model on the free API
  tier (override with `GENERATION_MODEL`). A Gemini app subscription doesn't
  include API access. Free-tier prompts may be used by Google to improve its
  products, so use a paid key for confidential deal documents.
- **Demo keyword lists** in `demo.py` decide which sentences count as risks,
  growth or overview. They are hand-written and English-only.
- **Number check is strict:** `$3.2 billion` fails if the source says
  `3,200 million`. Safer for diligence; loosen if it drops too much.
- **Rating scale:** one low/medium/high field, meaning severity for risks,
  impact for growth, importance for the summary.
- **Claim caps:** 8 risks, 6 growth, 6 summary points.
- **Executive summary** is generated from chunks directly, not from the other
  two reports.
