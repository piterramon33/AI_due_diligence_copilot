from app.generation import DemoLLM, ReportGenerator, SimpleChunk, get_llm

R = "doc_1111111111111111:c0001"
G = "doc_1111111111111111:c0002"
O = "doc_1111111111111111:c0003"

CHUNKS = [
    SimpleChunk(R, "Our top three customers accounted for 46% of revenue in 2024. "
                   "The loss of any of them could harm our results. "
                   "We also face litigation over a patent dispute.", {"title": "10-K"}),
    SimpleChunk(G, "Revenue grew 18.5% to $1,240 million. We plan to expand into Europe "
                   "through a new partnership with a distributor.", {"title": "Deck"}),
    SimpleChunk(O, "Acme designs and manufactures industrial sensors.\n"
                   "| Year | Revenue |\n| 2024 | 1,240 |", {"title": "10-K"}),
]


def test_factory_defaults_to_demo_without_keys(monkeypatch):
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(get_llm(), DemoLLM)


def test_demo_reports_pass_citation_checks():
    gen = ReportGenerator(DemoLLM())
    risk = gen.risk_assessment("Acme", CHUNKS)
    growth = gen.growth_opportunities("Acme", CHUNKS)
    summary = gen.executive_summary("Acme", CHUNKS)
    for report in (risk, growth, summary):
        assert report.claims, report
        assert report.rejected == []
    assert any("46%" in c.statement and c.citations[0].chunk_id == R for c in risk.claims)
    assert any("Europe" in c.statement and c.citations[0].chunk_id == G for c in growth.claims)
    assert any(c.category == "overview" for c in summary.claims)
    assert not any("|" in c.statement for r in (risk, growth, summary) for c in r.claims)


def test_demo_respects_claim_cap():
    report = ReportGenerator(DemoLLM()).risk_assessment("Acme", CHUNKS, max_claims=1)
    assert len(report.claims) == 1


def test_demo_with_no_matches_is_insufficient():
    chunks = [SimpleChunk("doc_2222222222222222:c0001", "The weather was pleasant on Tuesday afternoon.", {})]
    report = ReportGenerator(DemoLLM()).risk_assessment("Acme", chunks)
    assert report.insufficient_evidence and report.claims == []


def test_works_with_retrieval_output():
    from app.retrieval.models import RetrievedChunk

    rc = RetrievedChunk(
        chunk_id=R, document_id="doc_1111111111111111", text=CHUNKS[0].text,
        metadata={"title": "Acme 10-K", "doc_type": "filing", "page_start": 12,
                  "page_end": 12, "section": "Item 1A. Risk Factors"},
        rrf_score=0.03, rerank_score=4.2,
    )
    report = ReportGenerator(DemoLLM()).risk_assessment("Acme", [rc])
    cite = report.claims[0].citations[0]
    assert (cite.chunk_id, cite.page_start, cite.section) == (R, 12, "Item 1A. Risk Factors")
