"""Cited generation: risk assessment, growth opportunities, executive summary.

Usage:
    from app.generation import ReportGenerator, AnthropicLLM
    gen = ReportGenerator(AnthropicLLM())
    report = gen.risk_assessment("Acme Corp", retrieved_chunks)
    print(report.to_markdown())
"""

from .generator import ReportGenerator
from .llm import AnthropicLLM, FakeLLM, LLMOutputError, LLMRefusal, StructuredLLM
from .types import (
    ChunkLike,
    Citation,
    Claim,
    DraftClaim,
    DraftReport,
    RejectedClaim,
    Report,
    ReportKind,
    SimpleChunk,
)
from .validation import check_claim, check_claims

__all__ = [
    "AnthropicLLM", "ChunkLike", "Citation", "Claim", "DraftClaim", "DraftReport",
    "FakeLLM", "LLMOutputError", "LLMRefusal", "RejectedClaim", "Report",
    "ReportGenerator", "ReportKind", "SimpleChunk", "StructuredLLM",
    "check_claim", "check_claims",
]
