from app.ingestion import ingest_bytes
from app.ingestion.parsers import parse_html


def test_html_pages_from_css_breaks(sample_10k_html):
    parsed = parse_html(sample_10k_html)
    assert parsed.page_count == 4
    by_text = {b.text: b for b in parsed.blocks if b.type == "heading"}
    assert by_text["Item 1. Business"].page == 2
    assert by_text["Item 1A. Risk Factors"].page == 3
    assert by_text["Item 7. Management’s Discussion and Analysis"].page == 4


def test_html_hidden_xbrl_dropped(sample_10k_html):
    parsed = parse_html(sample_10k_html)
    assert "hidden xbrl facts" not in " ".join(b.text for b in parsed.blocks)


def test_html_financial_table_cleaned(sample_10k_html):
    parsed = parse_html(sample_10k_html)
    tables = [b for b in parsed.blocks if b.type == "table"]
    fin = tables[-1]
    assert fin.rows[1] == ["Revenue", "$1,250", "$1,100"]
    assert fin.page == 4


def test_html_ingest_sections(sample_10k_html, settings):
    doc, chunks = ingest_bytes(sample_10k_html, "acme-20251231.htm", settings=settings)
    assert doc.metadata["form_type"] == "10-K"
    risk = next(c for c in chunks if "supply chain" in c.text)
    assert risk.section_path == ["Part I", "Item 1A. Risk Factors"]
    assert risk.page_start == 3
    biz = next(c for c in chunks if "industrial sensors" in c.text)
    assert biz.section == "Item 1. Business"
    assert "Acme designs industrial sensors. It sells to manufacturers." in biz.text
