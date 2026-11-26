"""Integration tests against a real Postgres + pgvector (see conftest)."""

from app.retrieval import HybridRetriever, PgSearchStore, RetrievalConfig, SearchFilters
from app.retrieval.indexer import embed_pending_chunks


class KeywordOverlapReranker:
    """Stand-in cross-encoder: scores by how many query words a passage contains."""

    def __init__(self):
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        q = set(query.lower().split())
        return [float(len(q & set(p.lower().replace(",", " ").split()))) for p in passages]


def test_keyword_search_matches_partial_terms_and_section(seeded):
    store = PgSearchStore(seeded)
    hits = store.keyword_search("what is the customer concentration risk", limit=10)
    assert hits, "expected matches"
    assert hits[0].chunk_id.startswith("doc_acme10k:")
    assert "concentration" in hits[0].text.lower()
    # section heading is searchable too
    assert store.keyword_search("liquidity", limit=5)[0].metadata["section"].startswith("Liquidity")


def test_keyword_search_syntax_and_empty_queries(seeded):
    store = PgSearchStore(seeded)
    phrase = store.keyword_search('"credit facility"', limit=5)
    assert [h.chunk_id for h in phrase] == ["doc_acme10k:c0001"]
    excluded = store.keyword_search("customer -churn", limit=10)
    assert all("churn" not in h.text for h in excluded)
    assert store.keyword_search("the of and", limit=5) == []  # only stopwords
    assert store.keyword_search("!!! ) & |", limit=5) == []  # junk doesn't raise


def test_vector_search_and_filters(seeded, embedder):
    store = PgSearchStore(seeded)
    q = embedder.embed_query("revolving credit facility leverage covenant")
    hits = store.vector_search(q, limit=3)
    assert hits[0].chunk_id == "doc_acme10k:c0001"
    assert hits[0].score > hits[-1].score

    only_beta = store.vector_search(q, limit=10, filters=SearchFilters(companies=["Beta"]))
    assert only_beta and all(h.metadata["company"] == "Beta" for h in only_beta)
    decks = store.keyword_search(
        "revenue growth", limit=10, filters=SearchFilters(doc_types=["investor_presentation"])
    )
    assert [h.document_id for h in decks] == ["doc_acmedeck"]


def test_metadata_is_enriched_for_citations(seeded, embedder):
    store = PgSearchStore(seeded)
    hit = store.keyword_search("patent infringement", limit=1)[0]
    md = hit.metadata
    assert md["title"] == "doc_acme10k" and md["doc_type"] == "filing"
    assert md["page_start"] == 1 and md["section"] == "Legal Proceedings"
    assert md["chunk_index"] == 2 and md["kind"] == "text"


def test_hybrid_retrieve_fuses_and_reranks(seeded, embedder):
    reranker = KeywordOverlapReranker()
    retriever = HybridRetriever(
        PgSearchStore(seeded), embedder, reranker,
        RetrievalConfig(vector_k=5, keyword_k=5, rerank_top_n=4, top_k=2),
    )
    results = retriever.retrieve("customer churn pricing pressure from competitors")
    assert len(results) == 2
    assert results[0].chunk_id == "doc_betarpt:c0005"
    assert results[0].rerank_score >= results[1].rerank_score
    assert results[0].vector_rank is not None and results[0].keyword_rank is not None
    assert len(reranker.calls[0][1]) == 4  # only rerank_top_n candidates scored


def test_hybrid_without_reranker_returns_rrf_order(seeded, embedder):
    retriever = HybridRetriever(PgSearchStore(seeded), embedder, None, RetrievalConfig(top_k=3))
    results = retriever.retrieve("European expansion growth")
    assert results[0].document_id == "doc_acmedeck"
    assert all(r.rerank_score is None for r in results)
    assert [r.rrf_score for r in results] == sorted((r.rrf_score for r in results), reverse=True)
    assert retriever.retrieve("   ") == []


def test_min_rerank_score_drops_weak_chunks(seeded, embedder):
    retriever = HybridRetriever(
        PgSearchStore(seeded), embedder, KeywordOverlapReranker(),
        RetrievalConfig(min_rerank_score=2.0, top_k=10),
    )
    results = retriever.retrieve("patent class action Delaware")
    assert results and all(r.rerank_score >= 2.0 for r in results)


def test_indexer_is_idempotent_and_reembeds_on_model_change(seeded, embedder):
    assert embed_pending_chunks(seeded, embedder, model_name="hashing-test") == 0
    assert embed_pending_chunks(seeded, embedder, model_name="hashing-test-v2") == 6
