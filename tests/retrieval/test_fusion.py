import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import Candidate


def cand(cid: str, score: float = 0.0) -> Candidate:
    return Candidate(chunk_id=cid, document_id="doc_x", text=cid, metadata={}, score=score)


def test_rrf_scores_and_order():
    fused = reciprocal_rank_fusion(
        {"vector": [cand("a"), cand("b"), cand("c")], "keyword": [cand("c"), cand("a")]},
        k=60,
    )
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["a"].rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["c"].rrf_score == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["b"].rrf_score == pytest.approx(1 / 62)
    assert [c.chunk_id for c in fused] == ["a", "c", "b"]
    assert by_id["a"].vector_rank == 1 and by_id["a"].keyword_rank == 2
    assert by_id["b"].keyword_rank is None


def test_rrf_appearing_in_both_lists_beats_top_of_one():
    fused = reciprocal_rank_fusion(
        {"vector": [cand("only_vec"), cand("both")], "keyword": [cand("only_kw"), cand("both")]}
    )
    assert fused[0].chunk_id == "both"


def test_rrf_weights():
    lists = {"vector": [cand("v")], "keyword": [cand("k")]}
    assert reciprocal_rank_fusion(lists, weights={"keyword": 2.0})[0].chunk_id == "k"
    assert reciprocal_rank_fusion(lists, weights={"vector": 2.0})[0].chunk_id == "v"


def test_rrf_ignores_duplicates_within_a_list_and_handles_empty():
    fused = reciprocal_rank_fusion({"vector": [cand("a"), cand("a"), cand("b")], "keyword": []})
    assert {c.chunk_id: c.vector_rank for c in fused} == {"a": 1, "b": 2}
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []


def test_rrf_rejects_bad_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"vector": [cand("a")]}, k=0)
