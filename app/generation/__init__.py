"""Cited generation: risk assessment, growth opportunities, executive summary.

Usage:
    from app.generation import ReportGenerator, get_llm
    gen = ReportGenerator(get_llm())  # keyless demo unless a key is set
    report = gen.risk_assessment("Acme Corp", retrieved_chunks)
    print(report.to_markdown())
"""

from .demo import DemoLLM
from .generator import ReportGenerator
from .llm import AnthropicLLM, FakeLLM, GeminiLLM, get_llm, LLMOutputError, LLMRefusal, StructuredLLM
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
    "AnthropicLLM", "DemoLLM", "ChunkLike", "Citation", "Claim", "DraftClaim", "DraftReport",
    "FakeLLM", "GeminiLLM", "get_llm", "LLMOutputError", "LLMRefusal", "RejectedClaim", "Report",
    "ReportGenerator", "ReportKind", "SimpleChunk", "StructuredLLM",
    "check_claim", "check_claims",
]
