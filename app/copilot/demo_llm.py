"""Offline stand-in LLM: quotes the first clean sentence of the top sources.

It does no reasoning. It exists so the screen, retrieval and citation checks
can be tried without an API key: LLM_PROVIDER=demo.
"""

from __future__ import annotations

import re
from html import unescape

from app.generation.validation import split_sentences

_SOURCE_RE = re.compile(r'<source id="([^"]+)"[^>]*>\n(.*?)\n</source>', re.S)


class DemoLLM:
    model = "demo (offline, quotes sources)"

    def __init__(self, max_claims: int = 4) -> None:
        self.max_claims = max_claims

    def generate(self, *, system: str, prompt: str, schema: type):
        claims = []
        for chunk_id, body in _SOURCE_RE.findall(prompt):
            text = unescape(body).strip()
            if text.startswith("|"):  # Markdown table chunk
                continue
            sentence = next(
                (s.strip() for s in split_sentences(" ".join(text.split()))
                 if 40 <= len(s) <= 320 and s.rstrip().endswith(".")),
                None,
            )
            if not sentence:
                continue
            claims.append({
                "statement": f"{sentence} [{chunk_id}]",
                "citations": [chunk_id],
                "category": "quote",
                "rating": "medium",
            })
            if len(claims) >= self.max_claims:
                break
        return schema.model_validate({"claims": claims, "insufficient_evidence": not claims})
