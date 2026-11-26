"""Keyless demo provider: builds cited claims without any LLM or API.

It reads the <source> blocks from the prompt, scores each sentence with
keyword lists for the requested report, and returns the best sentences
verbatim with their chunk id as the citation. Because it only quotes the
sources, its claims pass the same citation checks as a real model's.

The output is extractive, so it reads like highlighted quotes rather than
analyst prose. It exists so the whole pipeline runs and demos with no key.
"""

from __future__ import annotations

import re
from html import unescape
from typing import TypeVar

from pydantic import BaseModel

from .types import DraftClaim, DraftReport

T = TypeVar("T", bound=BaseModel)

SOURCE_RE = re.compile(r'<source id="([^"]+)"[^>]*>\n(.*?)\n</source>', re.S)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# category -> keywords. Weight is the number of distinct keyword hits.
RISK_TERMS = {
    "liquidity": ["liquidity", "cash burn", "going concern", "covenant", "refinanc"],
    "leverage": ["debt", "leverage", "borrowing", "interest expense", "default"],
    "profitability": ["net loss", "decline", "decreased", "margin pressure", "impairment", "write-down"],
    "customer concentration": ["concentration", "top customer", "largest customer", "top three", "top five", "accounted for"],
    "supply chain": ["supplier", "supply chain", "shortage", "sole source", "single source"],
    "legal and regulatory": ["litigation", "lawsuit", "regulat", "investigation", "compliance", "fine", "penalt"],
    "competition": ["competition", "competitor", "competitive", "pricing pressure"],
    "operational": ["cybersecurity", "breach", "outage", "disruption", "key personnel"],
    "macro": ["inflation", "recession", "exchange rate", "currency", "tariff", "interest rate"],
    "general": ["risk", "uncertain", "adverse", "could harm", "may not"],
}
GROWTH_TERMS = {
    "revenue growth": ["revenue grew", "revenue increased", "growth of", "record revenue", "grew"],
    "market expansion": ["expand", "new market", "international", "europe", "asia", "addressable market", "tam"],
    "product": ["launch", "new product", "pipeline", "roadmap", "innovation"],
    "margin expansion": ["margin improved", "margin expansion", "cost savings", "efficienc"],
    "partnerships": ["partnership", "partner", "agreement", "acquisition", "joint venture"],
    "market trend": ["expected to grow", "cagr", "annually", "demand", "tailwind", "adoption"],
    "pricing": ["price increase", "pricing power", "upsell", "recurring revenue", "subscription"],
}
SUMMARY_TERMS = {
    "overview": ["we are", "the company", "provides", "designs", "manufactur", "headquartered", "operates"],
    "financials": ["revenue", "net income", "ebitda", "operating income", "cash flow", "gross margin"],
}


def _sentences(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("|")]
    out = []
    for sentence in SENTENCE_RE.split(" ".join(lines)):
        sentence = " ".join(sentence.split())
        if 30 <= len(sentence) <= 400 and sentence[-1] in ".!?":
            out.append(sentence)
    return out


def _score(sentence: str, terms: dict[str, list[str]]) -> tuple[int, str]:
    low = sentence.lower()
    best = (0, "")
    for category, words in terms.items():
        hits = sum(1 for w in words if w in low)
        if hits > best[0]:
            best = (hits, category)
    return best


def _rating(hits: int) -> str:
    return "high" if hits >= 3 else "medium" if hits == 2 else "low"


def _claim(sentence: str, chunk_id: str, category: str, hits: int) -> DraftClaim:
    body = sentence[:-1].rstrip()
    return DraftClaim(
        statement=f"{body} [{chunk_id}]{sentence[-1]}",
        citations=[chunk_id],
        category=category,
        rating=_rating(hits),
    )


def _ranked(sources, terms):
    ranked = []
    for order, (chunk_id, text) in enumerate(sources):
        for sentence in _sentences(text):
            hits, category = _score(sentence, terms)
            if hits:
                # Ties go to earlier sources, which retrieval ranked higher.
                ranked.append((-hits, order, sentence, chunk_id, category))
    ranked.sort(key=lambda r: (r[0], r[1]))
    return ranked


class DemoLLM:
    model = "demo-extractive"

    def generate(self, *, system: str, prompt: str, schema: type[T]) -> T:
        if schema is not DraftReport:
            raise TypeError("DemoLLM only produces DraftReport")
        sources = [(cid, unescape(text)) for cid, text in SOURCE_RE.findall(prompt)]
        task = prompt.rsplit("</sources>", 1)[-1].lower()
        limit_match = re.search(r"aim for (\d+) claims", task)
        limit = int(limit_match.group(1)) if limit_match else 6

        # A repair request means our quote failed a check. Quotes can't be
        # fixed by rewording, so drop them.
        if "failed the citation check" in task:
            return DraftReport(claims=[], insufficient_evidence=False)

        if "risk assessment" in task:
            rows = _ranked(sources, RISK_TERMS)
        elif "growth opportunities" in task:
            rows = _ranked(sources, GROWTH_TERMS)
        else:
            rows = self._summary_rows(sources)

        claims, seen = [], set()
        for neg_hits, _, sentence, chunk_id, category in rows:
            if sentence in seen:
                continue
            seen.add(sentence)
            claims.append(_claim(sentence, chunk_id, category, -neg_hits))
            if len(claims) >= limit:
                break
        return DraftReport(claims=claims, insufficient_evidence=not claims)

    @staticmethod
    def _summary_rows(sources):
        # One or two of each kind, so the summary covers the whole picture.
        picks = []
        for terms, label, take in (
            (SUMMARY_TERMS, None, 3),
            (RISK_TERMS, "risk", 2),
            (GROWTH_TERMS, "opportunity", 2),
        ):
            for row in _ranked(sources, terms)[:take]:
                neg, order, sentence, cid, category = row
                picks.append((neg, order, sentence, cid, label or category))
        return picks
