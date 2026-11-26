"""Plain text / Markdown parser (market reports, pasted notes). Form feeds (\\f) mark page breaks."""

from __future__ import annotations

import re

from app.ingestion.parsers.base import Block, ParsedDocument
from app.ingestion.sections import sec_heading

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")


def parse_text(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    pages = text.split("\f")
    blocks: list[Block] = []
    title: str | None = None
    for page_no, page in enumerate(pages, start=1):
        for para in re.split(r"\n\s*\n", page):
            lines = [ln.strip() for ln in para.strip().splitlines() if ln.strip()]
            buf: list[str] = []
            for ln in lines:
                heading = None
                if m := MD_HEADING_RE.match(ln):
                    # "#" -> level 3, "##" -> 4 ... so Markdown headings nest under SEC Part/Item headings.
                    heading = sec_heading(m.group(2)) or (len(m.group(1)) + 2, m.group(2))
                elif sec := sec_heading(ln):
                    heading = sec
                if heading:
                    if buf:
                        blocks.append(Block(type="text", text=" ".join(buf), page=page_no))
                        buf = []
                    level, label = heading
                    blocks.append(Block(type="heading", text=label, page=page_no, level=level))
                    title = title or label
                else:
                    buf.append(ln)
            if buf:
                blocks.append(Block(type="text", text=" ".join(buf), page=page_no))
    return ParsedDocument(blocks=blocks, page_count=len(pages), title=title)
