"""Format-independent parser output: an ordered list of blocks, each tied to a page."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

BlockType = Literal["heading", "text", "table"]


@dataclass
class Block:
    type: BlockType
    text: str
    page: int  # 1-based
    level: int = 0  # headings only: 1 = Part, 2 = Item, 3 = other heading
    rows: list[list[str]] | None = None  # tables only, cleaned cells


@dataclass
class ParsedDocument:
    blocks: list[Block]
    page_count: int
    title: str | None = None
    metadata: dict = field(default_factory=dict)


_WS = re.compile(r"\s+")


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    return _WS.sub(" ", str(value)).strip()


def clean_table(rows: list[list[object]]) -> list[list[str]]:
    """Normalise a raw table: tidy cells, glue lone currency symbols to their number, drop empty rows/cols."""
    cleaned: list[list[str]] = []
    for row in rows:
        cells = [clean_cell(c) for c in row]
        merged: list[str] = []
        skip_next = False
        for i, cell in enumerate(cells):
            if skip_next:
                skip_next = False
                merged.append("")
                continue
            # EDGAR and many PDFs put "$" (or "%"/")") in its own cell next to the number.
            if cell in {"$", "€", "£"} and i + 1 < len(cells) and cells[i + 1]:
                merged.append(f"{cell}{cells[i + 1]}")
                skip_next = True
            elif cell in {"%", ")"} and merged and merged[-1]:
                merged[-1] += cell
                merged.append("")
            else:
                merged.append(cell)
        if any(merged):
            cleaned.append(merged)
    if not cleaned:
        return []
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    keep = [j for j in range(width) if any(r[j] for r in cleaned)]
    return [[r[j] for j in keep] for r in cleaned]


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    esc = [[c.replace("|", "\\|") for c in r] for r in rows]
    header, body = esc[0], esc[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def table_block(rows: list[list[object]], page: int) -> Block | None:
    """Build a table block, or a text block for degenerate one-column/one-row layout tables."""
    cleaned = clean_table(rows)
    if not cleaned:
        return None
    if len(cleaned) < 2 or len(cleaned[0]) < 2:
        text = "\n".join(" ".join(c for c in r if c) for r in cleaned)
        return Block(type="text", text=text, page=page) if text.strip() else None
    return Block(type="table", text=table_to_markdown(cleaned), page=page, rows=cleaned)
