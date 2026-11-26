"""Upload-and-ask flow: keyword retrieval, Q&A and reports through the API with offline LLMs."""

from __future__ import annotations

import pytest

from app.api.copilot import get_ui_llm
from app.copilot.demo_llm import DemoLLM
from app.copilot.retrieve import KeywordRetriever, pool_for_report
from app.generation import FakeLLM, ReportKind
from app.storage import InMemoryStore
from app.ingestion import ingest_bytes
from app.models import DocType

TEXT_10K = b"""FORM 10-K
Acme Corp annual report.

# Item 1. Business
Acme designs industrial sensors sold to manufacturers in North America and Europe.

# Item 1A. Risk Factors
Our largest customer accounted for 31% of net revenue in fiscal 2025. Losing this customer would materially reduce revenue.
We rely on two contract manufacturers in Taiwan for all sensor assembly.

\f
# Item 7. Management's Discussion and Analysis
Net revenue grew 14% to $1,250 million, driven by demand from electric vehicle makers.
We plan to enter the Japanese market in 2026 through a distribution partnership.
"""


def _upload(client, data=TEXT_10K, name="acme-10k.txt", company="Acme"):
    r = client.post("/documents", files={"file": (name, data, "text/plain")},
                    data={"doc_type": "filing", "company": company})
    assert r.status_code == 201, r.text
    return r.json()["document"]["document_id"]


def test_keyword_retriever_ranks_and_filters(settings):
    store = InMemoryStore()
    doc, chunks = ingest_bytes(TEXT_10K, "acme-10k.txt", DocType.filing, "Acme", settings)
    store.save(doc, chunks)
    r = KeywordRetriever(store)
    hits = r.search("largest customer share of revenue", None, 3)
    assert hits and "largest customer" in hits[0].text
    assert r.search("largest customer", ["doc_other"], 3) == []
    pool = pool_for_report(r, ReportKind.RISK, None)
    assert len({c.chunk_id for c in pool}) == len(pool)


def test_keyword_retriever_sees_new_uploads(settings):
    store = InMemoryStore()
    r = KeywordRetriever(store)
    assert r.search("customer", None, 3) == []
    doc, chunks = ingest_bytes(TEXT_10K, "acme-10k.txt", DocType.filing, "Acme", settings)
    store.save(doc, chunks)
    assert r.search("customer", None, 3)


def test_ask_with_cited_answer(client):
    doc_id = _upload(client)

    def responder(system, prompt, schema):
        import re
        ids = re.findall(r'<source id="([^"]+)"', prompt)
        cid = next(i for i in ids if "c0" in i)
        return {"claims": [
            {"statement": f"The largest customer accounted for 31% of net revenue in fiscal 2025. [{ids[0]}]",
             "citations": [ids[0]], "category": "answer", "rating": "high"},
            {"statement": f"Revenue was $9,999 million. [{cid}]", "citations": [cid],
             "category": "answer", "rating": "high"},
        ], "insufficient_evidence": False}

    client.app.dependency_overrides[get_ui_llm] = lambda: FakeLLM(responder)
    r = client.post("/api/ask", json={"question": "How concentrated is revenue by customer?",
                                      "document_ids": [doc_id]})
    assert r.status_code == 200, r.text
    body = r.json()
    claims = body["answer"]["claims"]
    assert len(claims) == 1 and "31%" in claims[0]["statement"]
    cite = claims[0]["citations"][0]
    assert cite["source_title"] and cite["page_start"] == 1 and cite["section"] == "Item 1A. Risk Factors"
    # the invented number is dropped (once in the draft, once after the repair pass)
    assert body["answer"]["rejected"] and "9999" in body["answer"]["rejected"][0]["reasons"][0]
    assert body["sources"]

    chunk = client.get(f"/api/chunks/{cite['chunk_id']}").json()
    assert "31%" in chunk["text"]


@pytest.mark.parametrize("kind", [k.value for k in ReportKind])
def test_reports_with_demo_llm(client, kind):
    _upload(client)
    client.app.dependency_overrides[get_ui_llm] = lambda: DemoLLM()
    r = client.post("/api/reports", json={"kind": kind})
    assert r.status_code == 200, r.text
    report = r.json()["report"]
    assert report["company"] == "Acme"
    assert report["claims"], report
    for claim in report["claims"]:
        assert claim["citations"] and claim["citations"][0]["page_start"]


def test_missing_key_gives_clear_error(client, monkeypatch):
    _upload(client)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    import app.api.copilot as copilot
    monkeypatch.setattr(copilot, "_llm", None)
    r = client.post("/api/ask", json={"question": "What are the risks?"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]
    status = client.get("/api/status").json()
    assert status["llm_ready"] is False and status["retrieval"].startswith("keyword")


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Diligence Copilot" in r.text
