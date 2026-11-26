"""Offline stand-in for questions; reports use the generation package's DemoLLM.

The generation DemoLLM picks sentences with per-report keyword lists, which
suits reports but ignores the question. For a question this quotes the
source sentences that share the most words with it. No model, no API.
"""

from __future__ import annotations

from html import unescape

from app.generation import DemoLLM as ReportDemoLLM
from app.generation.demo import SOURCE_RE, _sentences
from app.generation.types import DraftClaim, DraftReport

from .retrieve import tokenize

QUESTION_MARKER = "Answer this question from the sources:"


class DemoLLM:
    model = "demo (offline, quotes sources)"

    def __init__(self, max_claims: int = 3) -> None:
        self.max_claims = max_claims
        self._reports = ReportDemoLLM()

    def generate(self, *, system: str, prompt: str, schema: type):
        task = prompt.rsplit("</sources>", 1)[-1]
        if QUESTION_MARKER not in task:
            return self._reports.generate(system=system, prompt=prompt, schema=schema)
        if "failed the citation check" in task:
            return DraftReport(claims=[], insufficient_evidence=False)
        question = task.split(QUESTION_MARKER, 1)[1].split("\n", 1)[0]
        terms = set(tokenize(question))
        rows = []
        for order, (chunk_id, body) in enumerate(SOURCE_RE.findall(prompt)):
            for sentence in _sentences(unescape(body)):
                hits = len(terms & set(tokenize(sentence)))
                if hits:
                    rows.append((-hits, order, sentence, chunk_id))
        rows.sort(key=lambda r: (r[0], r[1]))
        claims, seen = [], set()
        for neg, _, sentence, chunk_id in rows:
            if sentence in seen:
                continue
            seen.add(sentence)
            claims.append(DraftClaim(
                statement=f"{sentence[:-1].rstrip()} [{chunk_id}]{sentence[-1]}",
                citations=[chunk_id],
                category="answer" if not claims else "context",
                rating="high" if -neg >= 3 else "medium" if -neg == 2 else "low",
            ))
            if len(claims) >= self.max_claims:
                break
        return DraftReport(claims=claims, insufficient_evidence=not claims)
