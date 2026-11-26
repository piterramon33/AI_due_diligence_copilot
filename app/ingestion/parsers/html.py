"""HTML parser for SEC EDGAR filings (10-K, 10-Q, 8-K ...) and other HTML reports.

EDGAR HTML has no real pages; page breaks are CSS (page-break-before/after) on <hr> or <div> tags,
so we count those to recover the printed page number. Documents without them stay on page 1.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

from app.ingestion.parsers.base import Block, ParsedDocument, table_block
from app.ingestion.sections import sec_heading

BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "tr", "table", "ul", "ol",
    "section", "article", "blockquote", "pre", "center", "body", "br", "hr",
}
LEAF_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "center"}
SKIP_TAGS = {"script", "style", "head", "title", "meta", "noscript"}
PAGE_BREAK_RE = re.compile(r"(page-break-(before|after)|break-(before|after))\s*:\s*(always|page)", re.I)
HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)


def parse_html(data: bytes) -> ParsedDocument:
    # lxml may be missing; html.parser is slower but always there.
    soup = BeautifulSoup(data, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else None
    # Inline XBRL header (ix:header) carries hidden facts, not readable text.
    for hidden in soup.find_all(lambda t: t.name == "ix:header" or HIDDEN_RE.search(t.get("style", "") or "")):
        hidden.decompose()

    state = {"page": 1}
    blocks: list[Block] = []
    root = soup.body or soup
    _walk(root, blocks, state)
    page_count = state["page"]
    # A trailing break after the last content shouldn't create an empty page.
    if blocks:
        page_count = max(b.page for b in blocks)
    if not title:
        title = next((b.text for b in blocks if b.type == "heading"), None)
    return ParsedDocument(blocks=blocks, page_count=page_count, title=title)


def _breaks(tag: Tag, which: str) -> bool:
    style = tag.get("style", "") or ""
    return any(m.group(2) == which or m.group(4) == which for m in PAGE_BREAK_RE.finditer(style))


def _emit_text(text: str, blocks: list[Block], page: int, tag_name: str | None = None) -> None:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    if sec := sec_heading(text):
        blocks.append(Block(type="heading", text=sec[1], page=page, level=sec[0]))
    elif tag_name in {"h1", "h2", "h3", "h4"} and len(text) <= 120:
        blocks.append(Block(type="heading", text=text, page=page, level=3))
    else:
        blocks.append(Block(type="text", text=text, page=page))


def _row_cells(tr: Tag) -> list[str]:
    """Cells of a row with colspan expanded (text in the first column of the span) so columns line up."""
    cells: list[str] = []
    for cell in tr.find_all(["td", "th"]):
        cells.append(cell.get_text(" ", strip=True))
        try:
            span = int(cell.get("colspan", 1))
        except (TypeError, ValueError):
            span = 1
        cells.extend([""] * (min(span, 50) - 1))
    return cells


def _has_block_child(tag: Tag) -> bool:
    return any(isinstance(d, Tag) and d.name in BLOCK_TAGS for d in tag.descendants)


def _walk(node: Tag, blocks: list[Block], state: dict) -> None:
    inline_buffer: list[str] = []

    def flush_inline() -> None:
        if inline_buffer:
            _emit_text(" ".join(inline_buffer), blocks, state["page"])
            inline_buffer.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            if child.strip():
                inline_buffer.append(str(child))
            continue
        if not isinstance(child, Tag) or child.name in SKIP_TAGS:
            continue
        if child.name not in BLOCK_TAGS:  # inline element (span, b, a, ix:nonfraction ...)
            if _has_block_child(child):
                flush_inline()
                _walk(child, blocks, state)
            else:
                inline_buffer.append(child.get_text(" ", strip=True))
            continue

        flush_inline()
        if _breaks(child, "before"):
            state["page"] += 1
        if child.name == "table":
            rows = [_row_cells(tr) for tr in child.find_all("tr")]
            block = table_block(rows, state["page"])
            if block and block.type == "text":
                for line in block.text.splitlines():
                    _emit_text(line, blocks, state["page"])
            elif block:
                blocks.append(block)
        elif child.name in {"br", "hr"}:
            pass
        elif child.name in LEAF_BLOCK_TAGS and not _has_block_child(child):
            _emit_text(child.get_text(" ", strip=True), blocks, state["page"], child.name)
        else:
            _walk(child, blocks, state)
        if _breaks(child, "after"):
            state["page"] += 1
    flush_inline()
