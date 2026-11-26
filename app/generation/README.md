# Cited generation

Turns retrieved chunks into three reports: risk assessment, growth opportunities,
and executive summary. Every claim cites chunk IDs inline, like
`Top three customers are 46% of revenue [doc_ab12cd34ef567890:c0007].`
Claims that fail the source check are dropped from the report and kept in
`report.rejected` with the reasons.

```python
from app.generation import AnthropicLLM, ReportGenerator

gen = ReportGenerator(AnthropicLLM())          # reads ANTHROPIC_API_KEY
report = gen.risk_assessment("Acme Corp", chunks)  # chunks from retrieval
print(report.to_markdown())
```

## What gets rejected

- no inline `[chunk_id]` citation, or a sentence without one
- a cited ID that was not in the chunks given to the model
- a `citations` list that names IDs not cited inline
- any number not written in one of the chunks that claim cites
- a rating other than low / medium / high

Failed claims get one repair pass (`max_repairs=1`), then are dropped.

## Guesses to confirm

- **Provider:** Claude (`claude-opus-5-5`, override with `GENERATION_MODEL`)
  behind the `StructuredLLM` interface in `llm.py`. Another provider is one
  new class there.
- **Number check is strict:** `$3.2 billion` fails if the source says
  `3,200 million`. Safer for diligence; loosen if it drops too much.
- **Rating scale:** one low/medium/high field, meaning severity for risks,
  impact for growth, importance for the summary.
- **Claim caps:** 8 risks, 6 growth, 6 summary points.
- **Executive summary** is generated from chunks directly, not from the other
  two reports.
