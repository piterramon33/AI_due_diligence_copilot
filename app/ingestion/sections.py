"""Heading detection shared by all parsers, with SEC filing structure (PART / ITEM) built in."""

from __future__ import annotations

import re

# "PART II", "Part I — Financial Information"
PART_RE = re.compile(r"^\s*PART\s+([IVX]{1,4})\b\.?\s*[-–—:.]?\s*(.*)$", re.IGNORECASE)
# "Item 1A. Risk Factors", "ITEM 7 - MANAGEMENT'S DISCUSSION ...", "Item 9B."
ITEM_RE = re.compile(r"^\s*ITEM\s+(\d{1,2}[A-C]?)\s*[.:\-–—]?\s*(.*)$", re.IGNORECASE)

MAX_HEADING_CHARS = 120

FORM_RE = re.compile(r"\bFORM\s+(10-K|10-Q|8-K|20-F|40-F|S-1|S-4|F-1|DEF\s*14A)\b", re.IGNORECASE)


def sec_heading(line: str) -> tuple[int, str] | None:
    """Return (level, normalised label) if the line is a PART or ITEM heading."""
    text = line.strip()
    if not text or len(text) > MAX_HEADING_CHARS:
        return None
    # Body sentences that merely start with "Item 7 ..." usually run long and end with a period mid-text;
    # the length cap above plus requiring a short remainder keeps false positives low (guess, tune on real filings).
    if m := PART_RE.match(text):
        rest = m.group(2).strip()
        label = f"Part {m.group(1).upper()}" + (f" - {_title(rest)}" if rest else "")
        return 1, label
    if m := ITEM_RE.match(text):
        rest = m.group(2).strip().rstrip(".")
        if len(rest.split()) > 14:
            return None
        label = f"Item {m.group(1).upper()}." + (f" {_title(rest)}" if rest else "")
        return 2, label
    return None


def _title(s: str) -> str:
    # Filings often shout headings; make them readable but leave mixed-case text alone.
    return s.title().replace("'S ", "'s ") if s.isupper() else s


def detect_form_type(text: str) -> str | None:
    m = FORM_RE.search(text[:5000])
    return re.sub(r"\s+", " ", m.group(1).upper()) if m else None


def assign_sections(blocks):
    """Yield (block, section_path) for each non-heading block, maintaining a heading stack."""
    stack: list[tuple[int, str]] = []
    for block in blocks:
        if block.type == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            continue
        yield block, [label for _, label in stack]
