"""Reject claims that are not backed by the retrieved sources.

A claim survives only if all of these hold:

1. It has at least one inline citation marker like ``[doc_ab12...:c0007]``.
2. Every sentence in it carries its own citation marker.
3. Every cited chunk_id is one of the chunks actually given to the model
   (no invented or out-of-context sources).
4. The ``citations`` list and the inline markers agree.
5. Every number in the claim (percentages, amounts, years) appears in at
   least one of the chunks it cites.
6. The rating is one of low / medium / high.

Rule 5 is a deliberately strict lexical check. GUESS: it will reject correct
paraphrases such as "$3.2 billion" when the source says "3,200 million". That
is the safer failure mode for due diligence, and it can be loosened later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Mapping

from .types import ChunkLike, DraftClaim

# Anything in square brackets without spaces is treated as a citation marker.
# Unknown ids inside such markers are rejected rather than ignored.
CITATION_RE = re.compile(r"\[([^\[\]\s]+)\]")

# A number: optional sign, digits with optional thousands commas, optional decimals.
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")

# Split after sentence punctuation when the next sentence starts with a capital
# or a digit. A marker placed after the period stays with its sentence.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

VALID_RATINGS = {"low", "medium", "high"}


@dataclass
class ClaimCheck:
    claim: DraftClaim
    cited_ids: list[str]
    reasons: list[str]

    @property
    def ok(self) -> bool:
        return not self.reasons


def extract_citations(text: str) -> list[str]:
    """Inline citation ids in order of first appearance."""
    seen: dict[str, None] = {}
    for match in CITATION_RE.finditer(text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def strip_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text)


def _normalize_number(raw: str) -> str | None:
    cleaned = raw.replace(",", "").lstrip("+")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = value.normalize()
    # Decimal("1E+3") -> "1000"
    return format(normalized, "f")


def extract_numbers(text: str) -> set[str]:
    numbers = set()
    for match in NUMBER_RE.finditer(text):
        norm = _normalize_number(match.group(0))
        if norm is not None:
            numbers.add(norm.lstrip("-"))
    return numbers


def split_sentences(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def check_claim(claim: DraftClaim, chunks_by_id: Mapping[str, ChunkLike]) -> ClaimCheck:
    reasons: list[str] = []
    statement = claim.statement.strip()
    inline_ids = extract_citations(statement)

    if not statement:
        reasons.append("empty statement")

    if not inline_ids:
        reasons.append("no inline citation")

    for sentence in split_sentences(statement):
        if not extract_citations(sentence):
            reasons.append(f"sentence without citation: {sentence.strip()[:80]!r}")

    unknown = [cid for cid in inline_ids + list(claim.citations) if cid not in chunks_by_id]
    for cid in dict.fromkeys(unknown):
        reasons.append(f"cites unknown chunk {cid!r}")

    listed = set(claim.citations)
    inline = set(inline_ids)
    if listed - inline:
        reasons.append(
            "citations list has ids not cited inline: " + ", ".join(sorted(listed - inline))
        )

    known_cited = [cid for cid in inline_ids if cid in chunks_by_id]
    if known_cited:
        source_numbers: set[str] = set()
        for cid in known_cited:
            source_numbers |= extract_numbers(chunks_by_id[cid].text)
        missing = extract_numbers(strip_citations(statement)) - source_numbers
        for number in sorted(missing):
            reasons.append(f"number {number} not found in cited sources")

    if claim.rating.strip().lower() not in VALID_RATINGS:
        reasons.append(f"invalid rating {claim.rating!r}")

    return ClaimCheck(claim=claim, cited_ids=known_cited, reasons=reasons)


def check_claims(
    claims: Iterable[DraftClaim], chunks: Iterable[ChunkLike]
) -> list[ClaimCheck]:
    chunks_by_id = {c.chunk_id: c for c in chunks}
    return [check_claim(claim, chunks_by_id) for claim in claims]
