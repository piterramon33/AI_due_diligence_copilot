"""The bundled Apple 10-K preloads, and the offline demo answers numeric questions from its tables."""

from __future__ import annotations

import pytest

from app.copilot.demo_llm import DemoLLM, table_rows
from app.copilot.qa import answer_question
from app.copilot.retrieve import KeywordRetriever
from app.demo import DEMO_FILINGS, SAMPLE_QUESTIONS, preload
from app.storage import InMemoryStore


@pytest.fixture(scope="module")
def demo_store():
    if not DEMO_FILINGS[0].path.exists():
        pytest.skip("demo filing not bundled")
    store = InMemoryStore()
    assert len(preload(store)) == 1
    return store


def test_preload_is_idempotent_and_labelled(demo_store):
    assert preload(demo_store) == []
    (doc,) = demo_store.list_documents()
    assert doc.company == "Apple Inc."
    assert doc.metadata["demo"] is True
    assert doc.metadata["source_url"].startswith("https://www.sec.gov/")
    assert doc.metadata.get("form_type") == "10-K"
    sections = {c.section for c in demo_store.get_chunks(doc.document_id)}
    assert "Item 1A. Risk Factors" in sections


def test_table_rows_glue_currency_and_percent_cells():
    md = "|  | 2026 | Change |  | 2025 |\n| --- | --- | --- | --- | --- |\n| iPhone | $254,098 | 21 | % | $209,586 |"
    assert table_rows(md) == ["iPhone: 2026: $254,098; Change: 21%; 2025: $209,586."]


def test_demo_answers_net_sales_from_the_table(demo_store):
    retriever = KeywordRetriever(demo_store)
    question = "What were total net sales and how did they change from last year?"
    answer = answer_question(DemoLLM(), question, retriever.search(question, None, 8))
    assert answer.claims, answer
    assert "478,834" in answer.claims[0].statement
    assert not answer.rejected


@pytest.mark.parametrize("question", SAMPLE_QUESTIONS)
def test_every_sample_question_gets_a_cited_answer(demo_store, question):
    retriever = KeywordRetriever(demo_store)
    answer = answer_question(DemoLLM(), question, retriever.search(question, None, 8))
    assert answer.claims and all(c.citations for c in answer.claims)


def test_public_demo_caps_uploads_and_protects_the_sample(client, monkeypatch):
    from app.api.documents import get_store as dep  # same dependency the client overrides

    store = client.app.dependency_overrides[dep]()
    preload(store)
    sample = store.list_documents()[0].document_id
    monkeypatch.setenv("DDC_PUBLIC_DEMO", "1")
    monkeypatch.setenv("DDC_MAX_UPLOADS", "2")
    for i in range(3):
        r = client.post("/documents", files={"file": (f"n{i}.txt", f"# Note {i}\nRevenue grew {i}%.".encode(), "text/plain")})
        assert r.status_code == 201, r.text
    names = [d.filename for d in store.list_documents()]
    assert names.count("aapl-20260926.htm") == 1 and len(names) == 3 and "n0.txt" not in names
    assert client.delete(f"/documents/{sample}").status_code == 403
    assert client.get("/api/status").json()["public_demo"] is True
