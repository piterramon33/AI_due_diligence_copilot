"""Turn parsed blocks into citation-ready chunks.

Rules:
- A chunk never crosses a section boundary, so every chunk has exactly one section.
- Text chunks pack whole paragraphs up to `max_words`; an oversized paragraph is split by words.
- Consecutive text chunks in the same section overlap by `overlap_words`.
- Each table is its own chunk (Markdown). Big tables are split by rows, repeating the header row.
- page_start / page_end are exact: every word remembers the page it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers.base import Block, table_to_markdown
from app.ingestion.sections import assign_sections


@dataclass
class DraftChunk:
    text: str
    kind: str
    page_start: int
    page_end: int
    section_path: list[str]


Word = tuple[str, int]  # (word, page)


def chunk_blocks(blocks: list[Block], max_words: int = 350, overlap_words: int = 50) -> list[DraftChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    overlap_words = max(0, min(overlap_words, max_words // 2))
    out: list[DraftChunk] = []
    paras: list[list[Word]] = []  # pending paragraphs for the current section
    carried = 0  # how many pending words are overlap copied from the previous chunk
    current_path: list[str] | None = None

    def pending() -> int:
        return sum(len(p) for p in paras)

    def flush(carry_overlap: bool) -> None:
        nonlocal paras, carried
        if pending() > carried:  # never emit a chunk made only of overlap
            words = [w for p in paras for w in p]
            out.append(_make_text_chunk(paras, current_path or []))
            if carry_overlap and overlap_words and len(words) > overlap_words:
                paras, carried = [words[-overlap_words:]], overlap_words
                return
        paras, carried = [], 0

    for block, path in assign_sections(blocks):
        if path != current_path:
            flush(carry_overlap=False)
            current_path = path
        if block.type == "table":
            flush(carry_overlap=False)
            out.extend(_table_chunks(block, path, max_words))
            continue
        words: list[Word] = [(w, block.page) for w in block.text.split()]
        while words:
            room = max_words - pending()
            if len(words) <= room:
                paras.append(words)
                break
            if pending() > carried and len(words) <= max_words - overlap_words:
                # The paragraph fits in a fresh chunk: close this one instead of splitting the paragraph.
                flush(carry_overlap=True)
                continue
            if room > 0:
                paras.append(words[:room])
                words = words[room:]
            flush(carry_overlap=True)
    flush(carry_overlap=False)
    return out


def _make_text_chunk(paras: list[list[Word]], path: list[str]) -> DraftChunk:
    pages = [pg for p in paras for _, pg in p]
    text = "\n\n".join(" ".join(w for w, _ in p) for p in paras if p)
    return DraftChunk(text=text, kind="text", page_start=min(pages), page_end=max(pages), section_path=list(path))


def _table_chunks(block: Block, path: list[str], max_words: int) -> list[DraftChunk]:
    rows = block.rows or []
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    limit = max_words * 2  # tables are dense; allow them to run longer than prose (guess)
    chunks: list[DraftChunk] = []
    batch: list[list[str]] = []
    size = _words(header)
    for row in body:
        n = _words(row)
        if batch and size + n > limit:
            chunks.append(_table_chunk([header, *batch], block.page, path))
            batch, size = [], _words(header)
        batch.append(row)
        size += n
    if batch or not chunks:
        chunks.append(_table_chunk([header, *batch], block.page, path))
    return chunks


def _table_chunk(rows: list[list[str]], page: int, path: list[str]) -> DraftChunk:
    return DraftChunk(text=table_to_markdown(rows), kind="table", page_start=page, page_end=page, section_path=list(path))


def _words(row: list[str]) -> int:
    return sum(len(c.split()) for c in row)
