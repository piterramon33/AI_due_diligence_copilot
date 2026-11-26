"""Answer a free-form question with cited, checked claims.

Reuses the generation package's system prompt, source rendering, claim
validator and repair prompt, so answers follow the same citation rules as
the three reports. Only the task text differs.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, Field

from app.generation import Claim, DraftReport, RejectedClaim, StructuredLLM
from app.generation.prompts import SYSTEM_PROMPT, build_repair_prompt, render_sources
from app.generation.types import ChunkLike, Citation
from app.generation.validation import check_claims

MAX_ANSWER_CLAIMS = 5  # GUESS

QA_TASK = """\
Answer this question from the sources: {question}

Write the answer as a few claims, most direct first. Each claim is one or two \
sentences. Set category to "answer" for claims that answer the question and \
"context" for needed background. Set rating to how directly the sources \
answer the question (low, medium, high). Aim for {max_claims} claims or fewer. \
If the sources do not answer the question, set insufficient_evidence to true."""


class Answer(BaseModel):
    question: str
    claims: list[Claim]
    rejected: list[RejectedClaim] = Field(default_factory=list)
    insufficient_evidence: bool = False
    model: str | None = None


def answer_question(
    llm: StructuredLLM, question: str, chunks: Sequence[ChunkLike], *, max_repairs: int = 1
) -> Answer:
    chunks = list({c.chunk_id: c for c in chunks}.values())
    if not chunks:
        return Answer(question=question, claims=[], insufficient_evidence=True, model=llm.model)

    prompt = f"{render_sources(chunks)}\n\n" + QA_TASK.format(
        question=question.strip(), max_claims=MAX_ANSWER_CLAIMS
    )
    draft = llm.generate(system=SYSTEM_PROMPT, prompt=prompt, schema=DraftReport)
    checks = check_claims(draft.claims, chunks)
    passed = [c for c in checks if c.ok]
    failed = [c for c in checks if not c.ok]
    rejected = []
    for _ in range(max_repairs):
        if not failed:
            break
        repaired = llm.generate(
            system=SYSTEM_PROMPT,
            prompt=build_repair_prompt(prompt, [(c.claim, c.reasons) for c in failed]),
            schema=DraftReport,
        )
        recheck = check_claims(repaired.claims, chunks)
        passed += [c for c in recheck if c.ok]
        rejected += failed
        failed = [c for c in recheck if not c.ok]
    rejected += failed

    by_id = {c.chunk_id: c for c in chunks}
    seen: set[str] = set()
    claims = []
    for c in passed:
        key = " ".join(c.claim.statement.lower().split())
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            Claim(
                statement=c.claim.statement.strip(),
                citations=[Citation.from_chunk(by_id[cid]) for cid in c.cited_ids],
                category=c.claim.category.strip(),
                rating=c.claim.rating.strip().lower(),
            )
        )
    return Answer(
        question=question,
        claims=claims[:MAX_ANSWER_CLAIMS],
        rejected=[RejectedClaim(statement=c.claim.statement, reasons=c.reasons) for c in rejected],
        insufficient_evidence=not claims,
        model=llm.model,
    )
