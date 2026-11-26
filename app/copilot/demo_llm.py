"""Offline stand-in for questions; reports use the generation package's DemoLLM.

The generation DemoLLM picks sentences with per-report keyword lists, which
suits reports but ignores the question. For a question this quotes the
source sentences that share the most words with it. Rows of financial
tables count as sentences too ("Total net sales: 2026: $478,834; ..."), so
"how much" questions can be answered from the statements. No model, no API.
"""

from __future__ import annotations

import re
from html import unescape

from app.generation import DemoLLM as ReportDemoLLM
from app.generation.demo import SOURCE_RE, _sentences
from app.generation.types import DraftClaim, DraftReport

from .retrieve import tokenize

QUESTION_MARKER = "Answer this question from the sources:"
QUANT_RE = re.compile(r"\b(how much|how many|what (was|were|is|are) (the )?(total|amount)|total|amount|spend|spent|revenue|sales|income|margin|cash)\b", re.I)
ROWS_PER_TABLE = 2
HEADLINE_ROWS = ["total net sales", "net sales", "revenue", "total revenue", "gross margin", "operating income", "net income"]


def _cells(line: str) -> list[str]:
    """Split a Markdown table row and glue '$' and '%' cells back onto their numbers."""
    raw = [c.strip() for c in line.strip().strip("|").split("|")]
    out: list[str] = []
    pending = ""
    for cell in raw:
        if cell in {"$", "("}:
            pending += cell
        elif cell in {"%", ")"} and out:
            out[-1] += cell
        elif cell:
            out.append(pending + cell)
            pending = ""
        else:
            out.append("")
    return out


def table_rows(text: str) -> list[str]:
    """Turn Markdown table rows into one readable line each: "Label: col: value; col: value."."""
    header: list[str] = []
    rows = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|") or set(line.replace("|", "").strip()) <= {"-", " ", ":"}:
            continue
        cells = _cells(line)
        label, values = cells[0], [c for c in cells[1:] if c]
        if not values:
            continue
        if not label:
            header = values
            continue
        if not any(ch.isdigit() for ch in "".join(values)):
            continue
        pairs = [f"{h}: {v}" for h, v in zip(header, values)] if len(header) == len(values) else values
        line_out = f"{label.rstrip(':')}: {'; '.join(pairs)}."
        rows.append(re.sub(r"\(\s*([^()]*?)\s*\)", r"(\1)", line_out))
    return rows


class DemoLLM:
    model = "demo (offline, quotes sources)"

    def __init__(self, max_claims: int = 3) -> None:
        self.max_claims = max_claims
        self._reports = ReportDemoLLM()

    def generate(self, *, system: str, prompt: str, schema: type):
        task = prompt.rsplit("</sources>", 1)[-1]
        if QUESTION_MARKER not in task:
            report = self._reports.generate(system=system, prompt=prompt, schema=schema)
            if "executive summary" in task.lower() and "failed the citation check" not in task:
                report = self._add_headline_numbers(prompt, report)
            return report
        if "failed the citation check" in task:
            return DraftReport(claims=[], insufficient_evidence=False)
        question = task.split(QUESTION_MARKER, 1)[1].split("\n", 1)[0]
        terms = set(tokenize(question))
        quantitative = bool(QUANT_RE.search(question))
        rows = []
        for order, (chunk_id, body) in enumerate(SOURCE_RE.findall(prompt)):
            body = unescape(body)
            scored = []
            for sentence in _sentences(body):
                if " | " in sentence:  # a running page header glued to the sentence
                    sentence = sentence.rsplit(" | ", 1)[1].split(" ", 1)[-1]
                hits = len(terms & set(tokenize(sentence)))
                if hits and quantitative and any(ch.isdigit() for ch in sentence):
                    hits += 1
                if hits:
                    scored.append((-hits, order, sentence, chunk_id))
            table = []
            for sentence in table_rows(body):
                label = set(tokenize(sentence.split(":", 1)[0]))
                # The row label carries the meaning: reward words it shares with the
                # question, penalize extra words ("Percentage of total net sales").
                matched = len(terms & label)
                hits = 2 * matched - len(label - terms) + (1 if quantitative else 0)
                if matched >= 1 and hits >= 2 and quantitative:
                    table.append((-hits, order, sentence, chunk_id))
            table.sort()
            rows += scored + table[:ROWS_PER_TABLE]
        rows.sort(key=lambda r: (r[0], r[1]))
        claims, seen = [], set()
        for neg, _, sentence, chunk_id in rows:
            key = sentence.split(":", 1)[0] if ": " in sentence and sentence.count(";") else sentence
            if key in seen:  # the same table row also appears in another statement
                continue
            seen.add(key)
            claims.append(DraftClaim(
                statement=f"{sentence[:-1].rstrip()} [{chunk_id}]{sentence[-1]}",
                citations=[chunk_id],
                category="answer" if not claims else "context",
                rating="high" if -neg >= 3 else "medium" if -neg == 2 else "low",
            ))
            if len(claims) >= self.max_claims:
                break
        return DraftReport(claims=claims, insufficient_evidence=not claims)

    @staticmethod
    def _add_headline_numbers(prompt: str, report: DraftReport) -> DraftReport:
        """Lead an offline executive summary with the headline lines of the income statement."""
        found: dict[str, DraftClaim] = {}
        for chunk_id, body in SOURCE_RE.findall(prompt):
            for row in table_rows(unescape(body)):
                label = row.split(":", 1)[0].strip().lower()
                if label in HEADLINE_ROWS and label not in found:
                    found[label] = DraftClaim(
                        statement=f"{row[:-1]} [{chunk_id}].", citations=[chunk_id], category="financials", rating="high"
                    )
        headline = [found[k] for k in HEADLINE_ROWS if k in found]
        if not headline:
            return report
        claims = headline + report.claims[: max(0, len(report.claims) - len(headline) + 2)]
        return DraftReport(claims=claims, insufficient_evidence=False)

