"""Shared data model. Other packages (retrieval, generation) import these instead of defining their own."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocType(str, Enum):
    filing = "filing"  # 10-K, 10-Q, 8-K, S-1, 20-F ... (form in metadata["form_type"])
    financial_statement = "financial_statement"
    investor_presentation = "investor_presentation"
    market_report = "market_report"
    other = "other"


BlockKind = Literal["text", "table"]


class Document(BaseModel):
    document_id: str  # "doc_" + first 16 hex chars of the file's sha256
    filename: str
    title: str
    doc_type: DocType = DocType.other
    company: str | None = None
    sha256: str
    mime_type: str
    page_count: int
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str  # "<document_id>:c<4-digit index>"
    document_id: str
    chunk_index: int
    text: str
    kind: BlockKind = "text"  # tables are rendered as Markdown
    page_start: int  # 1-based
    page_end: int
    section: str | None = None  # innermost heading, e.g. "Item 1A. Risk Factors"
    section_path: list[str] = Field(default_factory=list)  # e.g. ["Part I", "Item 1A. Risk Factors"]
    metadata: dict[str, Any] = Field(default_factory=dict)  # source, title, doc_type, company, ...

    @staticmethod
    def make_id(document_id: str, index: int) -> str:
        return f"{document_id}:c{index:04d}"

    def citation(self) -> str:
        """Human-readable source label, e.g. 'acme-10k.pdf, p. 12-13, Item 1A. Risk Factors'."""
        source = self.metadata.get("source") or self.document_id
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"p. {self.page_start}-{self.page_end}"
        parts = [source, pages]
        if self.section:
            parts.append(self.section)
        return ", ".join(parts)
