"""Data shapes for cited generation.

Retrieval and ingestion own the real chunk model. This module only needs a few
attributes from it, so it accepts anything that matches `ChunkLike` instead of
importing a concrete class. When the shared chunk schema lands, nothing here
should need to change as long as it exposes `chunk_id`, `text` and `metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class ChunkLike(Protocol):
    """Minimal view of a retrieved chunk.

    Matches `app.models.Chunk` from the ingestion work: chunk_id looks like
    "doc_ab12cd34ef567890:c0007". Optional attributes we read when present:
    page_start, page_end, section, and metadata keys title, source, doc_type.
    """

    chunk_id: str
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SimpleChunk:
    """Concrete chunk used by tests and as a stand-in until ingestion ships."""

    chunk_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ReportKind(str, Enum):
    RISK = "risk_assessment"
    GROWTH = "growth_opportunities"
    SUMMARY = "executive_summary"


# --- What the LLM is asked to return (structured output schema) -------------


class DraftClaim(BaseModel):
    """One claim as the model drafts it."""

    statement: str = Field(
        description=(
            "One factual sentence. Every sentence must end with one or more "
            "inline citations like [chunk_id] naming the source chunks."
        )
    )
    citations: list[str] = Field(
        description="Every chunk_id cited inline in the statement, no others."
    )
    category: str = Field(
        description=(
            "Short label, e.g. 'liquidity', 'customer concentration', "
            "'regulatory', 'market expansion'. Use 'overview' for summaries."
        )
    )
    # GUESS: a 3-level scale is enough for a first version. Risk claims use it
    # for severity, growth claims for potential impact, summaries for salience.
    rating: str = Field(description="One of: low, medium, high.")


class DraftReport(BaseModel):
    claims: list[DraftClaim]
    insufficient_evidence: bool = Field(
        description=(
            "True when the sources do not support any claim for this report. "
            "Then return an empty claims list."
        )
    )


# --- What callers get back ---------------------------------------------------


class Citation(BaseModel):
    chunk_id: str
    source_title: str | None = None
    doc_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None

    @classmethod
    def from_chunk(cls, chunk: ChunkLike) -> "Citation":
        meta = dict(getattr(chunk, "metadata", None) or {})
        return cls(
            chunk_id=chunk.chunk_id,
            source_title=meta.get("title") or meta.get("source"),
            doc_type=meta.get("doc_type"),
            page_start=getattr(chunk, "page_start", None) or meta.get("page_start"),
            page_end=getattr(chunk, "page_end", None) or meta.get("page_end"),
            section=getattr(chunk, "section", None) or meta.get("section"),
        )


class Claim(BaseModel):
    statement: str
    citations: list[Citation]
    category: str
    rating: str


class RejectedClaim(BaseModel):
    statement: str
    reasons: list[str]


class Report(BaseModel):
    kind: ReportKind
    company: str
    claims: list[Claim]
    rejected: list[RejectedClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    model: str | None = None

    def to_markdown(self) -> str:
        title = self.kind.value.replace("_", " ").title()
        lines = [f"## {title}: {self.company}", ""]
        if not self.claims:
            lines.append("_Not enough sourced evidence to make any claim._")
        for claim in self.claims:
            lines.append(f"- **[{claim.rating}] {claim.category}**: {claim.statement}")
        return "\n".join(lines)
