from app.ingestion import ingest_bytes
from app.ingestion.parsers import parse_pdf


def test_pdf_blocks_keep_pages_and_detect_headings(sample_10k_pdf):
    parsed = parse_pdf(sample_10k_pdf)
    assert parsed.page_count == 3
    assert parsed.title == "Acme Corp 2025 10-K"
    headings = [(b.level, b.text, b.page) for b in parsed.blocks if b.type == "heading"]
    assert (1, "Part I", 1) in headings
    assert (2, "Item 1A. Risk Factors", 2) in headings
    assert (2, "Item 7. Management's Discussion and Analysis", 3) in headings
    assert not parsed.metadata["needs_ocr"]


def test_pdf_drops_running_footer_and_page_numbers(sample_10k_pdf):
    parsed = parse_pdf(sample_10k_pdf)
    texts = " ".join(b.text for b in parsed.blocks)
    assert "Acme Corp | 2025 Form 10-K" not in texts


def test_pdf_table_extracted_with_merged_rows(sample_10k_pdf):
    parsed = parse_pdf(sample_10k_pdf)
    tables = [b for b in parsed.blocks if b.type == "table"]
    assert len(tables) == 1
    assert tables[0].page == 3
    assert tables[0].rows[1] == ["Revenue", "$1,250", "$1,100"]
    # Table text must not leak into the surrounding prose.
    prose = " ".join(b.text for b in parsed.blocks if b.type == "text")
    assert "$1,250" not in prose


def test_pdf_ingest_chunks_carry_citation_metadata(sample_10k_pdf, settings):
    doc, chunks = ingest_bytes(sample_10k_pdf, "acme-10k.pdf", company="Acme", settings=settings)
    assert doc.doc_type.value == "filing"  # detected from "FORM 10-K"
    assert doc.metadata["form_type"] == "10-K"
    assert chunks and all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks)
    assert all(c.metadata["source"] == "acme-10k.pdf" and c.metadata["company"] == "Acme" for c in chunks)

    risk = [c for c in chunks if c.section == "Item 1A. Risk Factors"]
    assert len(risk) >= 2, "long risk section should split into several chunks"
    assert all(c.section_path == ["Part I", "Item 1A. Risk Factors"] for c in risk)
    assert all(c.page_start == 2 for c in risk)

    table = next(c for c in chunks if c.kind == "table")
    assert table.section_path == ["Part II", "Item 7. Management's Discussion and Analysis"]
    assert table.page_start == table.page_end == 3
    assert "| Revenue | $1,250 | $1,100 |" in table.text
    assert table.citation() == "acme-10k.pdf, p. 3, Item 7. Management's Discussion and Analysis"
    assert table.chunk_id == f"{doc.document_id}:c{table.chunk_index:04d}"


def test_scanned_pdf_flagged_for_ocr(settings):
    import io

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.rect(100, 100, 200, 200, fill=1)  # an "image" with no text layer
    c.showPage()
    c.save()
    doc, chunks = ingest_bytes(buf.getvalue(), "scan.pdf", settings=settings)
    assert doc.metadata["needs_ocr"] is True
    assert chunks == []
