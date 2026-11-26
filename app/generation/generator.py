"""Generate cited risk assessments, growth opportunities and executive summaries.

Flow for each report:
1. Render the retrieved chunks as <source id="..."> blocks.
2. Ask the LLM for structured claims with inline [chunk_id] citations.
3. Check every claim against the sources (see validation.py).
4. Send failed claims back once for repair, then check them again.
5. Return only claims that pass. Every draft that failed a check, including
   ones later fixed by the repair pass, is logged in `rejected` with reasons,
   so the UI can show what was dropped and why.
"""

from __future__ import annotations

from typing import Sequence

from .llm import StructuredLLM
from .prompts import SYSTEM_PROMPT, build_prompt, build_repair_prompt
from .types import (
    ChunkLike,
    Citation,
    Claim,
    DraftReport,
    RejectedClaim,
    Report,
    ReportKind,
)
from .validation import ClaimCheck, check_claims

# GUESS: defaults chosen for a first version, tune once real filings are in.
DEFAULT_MAX_CLAIMS = {
    ReportKind.RISK: 8,
    ReportKind.GROWTH: 6,
    ReportKind.SUMMARY: 6,
}
DEFAULT_MAX_REPAIRS = 1


class ReportGenerator:
    def __init__(self, llm: StructuredLLM, *, max_repairs: int = DEFAULT_MAX_REPAIRS) -> None:
        self.llm = llm
        self.max_repairs = max_repairs

    def generate(
        self,
        kind: ReportKind,
        company: str,
        chunks: Sequence[ChunkLike],
        *,
        max_claims: int | None = None,
    ) -> Report:
        chunks = _dedupe(chunks)
        if not chunks:
            return Report(kind=kind, company=company, claims=[], insufficient_evidence=True,
                          model=self.llm.model)

        max_claims = max_claims or DEFAULT_MAX_CLAIMS[kind]
        prompt = build_prompt(kind, company, chunks, max_claims)
        draft = self.llm.generate(system=SYSTEM_PROMPT, prompt=prompt, schema=DraftReport)

        checks = check_claims(draft.claims, chunks)
        passed = [c for c in checks if c.ok]
        failed = [c for c in checks if not c.ok]
        rejected_log: list[ClaimCheck] = []

        for _ in range(self.max_repairs):
            if not failed:
                break
            repair_prompt = build_repair_prompt(prompt, [(c.claim, c.reasons) for c in failed])
            repaired = self.llm.generate(
                system=SYSTEM_PROMPT, prompt=repair_prompt, schema=DraftReport
            )
            recheck = check_claims(repaired.claims, chunks)
            passed += [c for c in recheck if c.ok]
            rejected_log += failed
            failed = [c for c in recheck if not c.ok]

        passed = _dedupe_claims(passed)[:max_claims]
        by_id = {c.chunk_id: c for c in chunks}
        claims = [_to_claim(c, by_id) for c in passed]
        rejected = [
            RejectedClaim(statement=c.claim.statement, reasons=c.reasons)
            for c in rejected_log + failed
        ]
        return Report(
            kind=kind,
            company=company,
            claims=claims,
            rejected=rejected,
            insufficient_evidence=not claims,
            model=self.llm.model,
        )

    def risk_assessment(self, company: str, chunks: Sequence[ChunkLike], **kw) -> Report:
        return self.generate(ReportKind.RISK, company, chunks, **kw)

    def growth_opportunities(self, company: str, chunks: Sequence[ChunkLike], **kw) -> Report:
        return self.generate(ReportKind.GROWTH, company, chunks, **kw)

    def executive_summary(self, company: str, chunks: Sequence[ChunkLike], **kw) -> Report:
        return self.generate(ReportKind.SUMMARY, company, chunks, **kw)


def _dedupe(chunks: Sequence[ChunkLike]) -> list[ChunkLike]:
    seen: dict[str, ChunkLike] = {}
    for chunk in chunks:
        seen.setdefault(chunk.chunk_id, chunk)
    return list(seen.values())


def _dedupe_claims(checks: list[ClaimCheck]) -> list[ClaimCheck]:
    seen: dict[str, ClaimCheck] = {}
    for check in checks:
        key = " ".join(check.claim.statement.lower().split())
        seen.setdefault(key, check)
    return list(seen.values())


def _to_claim(check: ClaimCheck, by_id: dict[str, ChunkLike]) -> Claim:
    return Claim(
        statement=check.claim.statement.strip(),
        citations=[Citation.from_chunk(by_id[cid]) for cid in check.cited_ids],
        category=check.claim.category.strip(),
        rating=check.claim.rating.strip().lower(),
    )
