"""Glue between the UI and the ingestion, retrieval and generation packages.

- `retrieve.py`: picks a retriever. Hybrid (pgvector + full-text + rerank)
  when Postgres and the retrieval extras are available, otherwise an
  in-memory BM25 keyword search so the app runs with nothing installed.
- `qa.py`: answers a free-form question with the same cited-claim checks
  the reports use.
- `demo_llm.py`: an offline stand-in LLM that quotes sources, for trying
  the screen without an API key (LLM_PROVIDER=demo).
"""
