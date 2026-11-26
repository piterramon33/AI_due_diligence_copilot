"""PDF parser (pdfplumber): text lines grouped into paragraphs, ruled tables, font-size headings.

Works for filings, financial statements and investor decks that have a text layer.
Scanned PDFs come back with metadata["needs_ocr"] = True and few or no blocks; OCR is not done here yet.
"""

from __future__ import annotations

import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass

import pdfplumber

from app.ingestion.parsers.base import Block, ParsedDocument, table_block
from app.ingestion.sections import sec_heading

# Guess: a line whose font is >= 18% larger than body text is a heading. Tune on real decks/filings.
HEADING_SIZE_RATIO = 1.18
PAGE_NUMBER_RE = re.compile(r"^(page\s+)?\d{1,4}(\s+of\s+\d{1,4})?$|^[-–]\s*\d{1,4}\s*[-–]$", re.IGNORECASE)
MARGIN_FRACTION = 0.08  # top/bottom band where running headers and footers live


@dataclass
class _Line:
    text: str
    top: float
    bottom: float
    size: float
    bold: bool
    page: int
    in_margin: bool


def parse_pdf(data: bytes) -> ParsedDocument:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page_count = len(pdf.pages)
        title = (pdf.metadata or {}).get("Title") or None
        lines_by_page: list[list[_Line]] = []
        tables_by_page: list[list[tuple[float, Block]]] = []
        for page_no, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            bboxes = [t.bbox for t in tables]
            page_tables: list[tuple[float, Block]] = []
            for t in tables:
                block = table_block(t.extract(), page_no)
                if block:
                    page_tables.append((t.bbox[1], block))
            tables_by_page.append(page_tables)

            text_page = page.filter(lambda obj, b=bboxes: not _inside_any(obj, b)) if bboxes else page
            height = float(page.height) or 1.0
            lines: list[_Line] = []
            for raw in text_page.extract_text_lines(return_chars=True, strip=True):
                text = raw["text"].strip()
                chars = [c for c in raw.get("chars", []) if c.get("text", "").strip()]
                if not text or not chars:
                    continue
                size = statistics.median(float(c.get("size", 0)) for c in chars)
                bold = sum("bold" in str(c.get("fontname", "")).lower() for c in chars) > len(chars) / 2
                top, bottom = float(raw["top"]), float(raw["bottom"])
                in_margin = top < height * MARGIN_FRACTION or bottom > height * (1 - MARGIN_FRACTION)
                lines.append(_Line(text, top, bottom, size, bold, page_no, in_margin))
            lines_by_page.append(lines)

    all_lines = [ln for page in lines_by_page for ln in page]
    total_chars = sum(len(ln.text) for ln in all_lines)
    metadata: dict = {"needs_ocr": page_count > 0 and total_chars < 50 * page_count}
    if not all_lines and not any(tables_by_page):
        return ParsedDocument(blocks=[], page_count=page_count, title=title, metadata=metadata)

    body_size = _body_font_size(all_lines)
    boilerplate = _running_headers(lines_by_page)

    blocks: list[Block] = []
    for page_no, (lines, tables) in enumerate(zip(lines_by_page, tables_by_page), start=1):
        items: list[tuple[float, Block]] = list(tables)
        para: list[_Line] = []

        def flush() -> None:
            if para:
                items.append((para[0].top, Block(type="text", text=_join_lines(para), page=page_no)))
                para.clear()

        for ln in lines:
            if ln.in_margin and (PAGE_NUMBER_RE.match(ln.text) or _norm(ln.text) in boilerplate):
                continue
            heading = _classify_heading(ln, body_size)
            if heading:
                flush()
                level, label = heading
                items.append((ln.top, Block(type="heading", text=label, page=page_no, level=level)))
                continue
            if para and _starts_new_paragraph(para[-1], ln):
                flush()
            para.append(ln)
        flush()
        items.sort(key=lambda x: x[0])
        blocks.extend(b for _, b in items)

    if not title:
        title = next((b.text for b in blocks if b.type == "heading" and b.level == 3), None)
    return ParsedDocument(blocks=blocks, page_count=page_count, title=title, metadata=metadata)


def _inside_any(obj: dict, bboxes: list[tuple[float, float, float, float]]) -> bool:
    if "x0" not in obj:
        return False
    cx = (float(obj["x0"]) + float(obj["x1"])) / 2
    cy = (float(obj["top"]) + float(obj["bottom"])) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


def _body_font_size(lines: list[_Line]) -> float:
    # Weight by characters so a few big titles don't move the estimate.
    counts: Counter[float] = Counter()
    for ln in lines:
        counts[round(ln.size, 1)] += len(ln.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def _norm(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def _running_headers(lines_by_page: list[list[_Line]]) -> set[str]:
    """Margin lines repeated on at least half the pages (e.g. 'Acme Corp | 2025 Form 10-K | 17')."""
    pages = len(lines_by_page)
    if pages < 3:
        return set()
    seen: Counter[str] = Counter()
    for lines in lines_by_page:
        seen.update({_norm(ln.text) for ln in lines if ln.in_margin})
    return {text for text, n in seen.items() if n >= max(2, pages // 2)}


def _classify_heading(ln: _Line, body_size: float) -> tuple[int, str] | None:
    if sec := sec_heading(ln.text):
        return sec
    text = ln.text
    if len(text) > 100 or not re.search(r"[A-Za-z]", text) or text.endswith((".", ",", ";")):
        return None
    if ln.size >= body_size * HEADING_SIZE_RATIO:
        return 3, text
    return None


def _starts_new_paragraph(prev: _Line, cur: _Line) -> bool:
    line_height = max(prev.bottom - prev.top, 1.0)
    gap = cur.top - prev.bottom
    return gap > 0.6 * line_height or abs(cur.size - prev.size) > 1.0


def _join_lines(lines: list[_Line]) -> str:
    out = ""
    for ln in lines:
        if not out:
            out = ln.text
        elif out.endswith("-") and ln.text[:1].islower():
            out = out[:-1] + ln.text  # de-hyphenate "manage-\nment"
        else:
            out += " " + ln.text
    return out
