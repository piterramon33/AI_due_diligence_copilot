from app.generation import DraftClaim
from app.generation.validation import check_claims, extract_citations, extract_numbers

from .conftest import A, B, C


def claim(statement, citations, rating="high"):
    return DraftClaim(statement=statement, citations=citations, category="x", rating=rating)


def reasons(c, chunks):
    return check_claims([c], chunks)[0].reasons


def test_well_cited_claim_passes(chunks):
    c = claim(f"Top three customers make up 46% of revenue [{B}].", [B])
    assert reasons(c, chunks) == []


def test_marker_after_period_counts(chunks):
    c = claim(f"Revenue grew 18.5% in fiscal 2024. [{A}]", [A])
    assert reasons(c, chunks) == []


def test_uncited_claim_rejected(chunks):
    r = reasons(claim("Acme faces heavy competition.", []), chunks)
    assert "no inline citation" in r


def test_unknown_chunk_rejected(chunks):
    r = reasons(claim("Margins are thin [doc_zzzz:c9999].", ["doc_zzzz:c9999"]), chunks)
    assert any("unknown chunk" in x for x in r)


def test_every_sentence_needs_citation(chunks):
    c = claim(f"Revenue grew 18.5% [{A}]. Management expects this to continue.", [A])
    assert any("sentence without citation" in x for x in reasons(c, chunks))


def test_citation_list_must_match_inline(chunks):
    c = claim(f"Revenue grew 18.5% [{A}].", [A, B])
    assert any("not cited inline" in x for x in reasons(c, chunks))


def test_inline_ids_missing_from_list_are_still_used(chunks):
    # The inline markers are the source of truth, the list is a cross-check.
    c = claim(f"Revenue grew 18.5% [{A}] and customers are concentrated [{B}].", [A])
    assert check_claims([c], chunks)[0].ok
    assert check_claims([c], chunks)[0].cited_ids == [A, B]


def test_invented_number_rejected(chunks):
    c = claim(f"Revenue grew 25% in fiscal 2024 [{A}].", [A])
    assert "number 25 not found in cited sources" in reasons(c, chunks)


def test_number_from_uncited_chunk_rejected(chunks):
    # 46% is in chunk B, but the claim only cites A.
    c = claim(f"Customer concentration is 46% [{A}].", [A])
    assert "number 46 not found in cited sources" in reasons(c, chunks)


def test_number_formatting_normalized(chunks):
    c = claim(f"Revenue reached $1240 million [{A}].", [A])
    assert reasons(c, chunks) == []


def test_chunk_id_digits_not_treated_as_numbers(chunks):
    c = claim(f"The European sensor market should grow 9% a year [{C}].", [C])
    assert reasons(c, chunks) == []


def test_bad_rating_rejected(chunks):
    c = claim(f"Revenue grew 18.5% [{A}].", [A], rating="critical")
    assert any("invalid rating" in x for x in reasons(c, chunks))


def test_helpers():
    assert extract_citations("a [x:1] b [y] [x:1]") == ["x:1", "y"]
    assert extract_numbers("1,240 and 18.50% and 2024") == {"1240", "18.5", "2024"}
