"""Fixtures build small synthetic documents at test time so no real filings are committed."""

from __future__ import annotations

import io
import os

os.environ["DDC_DEMO_PRELOAD"] = "0"  # tests start from an empty store; tests/test_demo.py covers preloading

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.storage import InMemoryStore, get_store

LOREM = (
    "Demand for our products depends on enterprise IT budgets, which may decline in an economic downturn. "
    "We rely on a small number of contract manufacturers and any disruption could delay shipments. "
)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawString(72, 30, "Acme Corp | 2025 Form 10-K")
    canvas.drawRightString(540, 30, str(doc.page))
    canvas.restoreState()


@pytest.fixture(scope="session")
def sample_10k_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=13)
    title = ParagraphStyle("title", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=20, leading=24)
    item = ParagraphStyle("item", parent=body, fontName="Helvetica-Bold")

    story = [
        Paragraph("Acme Corp Annual Report", title),
        Spacer(1, 12),
        Paragraph("UNITED STATES SECURITIES AND EXCHANGE COMMISSION. FORM 10-K. Annual report.", body),
        Paragraph("PART I", item),
        Paragraph("Item 1. Business", item),
        Paragraph("Acme designs industrial sensors sold to manufacturers in North America and Europe. " * 4, body),
        PageBreak(),
        Paragraph("Item 1A. Risk Factors", item),
        *[Paragraph(LOREM * 3, body) for _ in range(6)],
        PageBreak(),
        Paragraph("PART II", item),
        Paragraph("Item 7. Management's Discussion and Analysis", item),
        Paragraph("Revenue grew on higher sensor volumes. The table below summarises results.", body),
        Spacer(1, 8),
        Table(
            [["(in millions)", "2025", "2024"], ["Revenue", "$1,250", "$1,100"], ["Net income", "$140", "$95"]],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]),
        ),
        Spacer(1, 8),
        Paragraph("Gross margin expanded due to pricing actions and lower freight costs.", body),
    ]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter, title="Acme Corp 2025 10-K").build(
        story, onFirstPage=_footer, onLaterPages=_footer
    )
    return buf.getvalue()


SAMPLE_HTML = b"""<html><head><title>acme-20251231</title></head><body>
<div style="display:none"><ix:header>hidden xbrl facts</ix:header></div>
<p style="text-align:center"><b>FORM 10-K</b></p>
<p>Acme Corp annual report for the fiscal year ended December 31, 2025.</p>
<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr><tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr></table>
<hr style="page-break-after:always"/>
<div><span style="font-weight:bold">PART I</span></div>
<div><span style="font-weight:bold">ITEM 1. BUSINESS</span></div>
<div><span>Acme designs industrial sensors.</span> <span>It sells to manufacturers.</span></div>
<div style="page-break-before:always"><span style="font-weight:bold">Item 1A. Risk Factors</span></div>
<p>Our supply chain depends on a few contract manufacturers.</p>
<hr style="page-break-after:always"/>
<div><b>PART II</b></div>
<div><b>Item 7. Management&#8217;s Discussion and Analysis</b></div>
<table>
  <tr><td></td><td colspan="2">2025</td><td colspan="2">2024</td></tr>
  <tr><td>Revenue</td><td>$</td><td>1,250</td><td>$</td><td>1,100</td></tr>
  <tr><td>Net income</td><td>$</td><td>140</td><td>$</td><td>95</td></tr>
</table>
<p>Margins improved.</p>
</body></html>"""


@pytest.fixture(scope="session")
def sample_10k_html() -> bytes:
    return SAMPLE_HTML


@pytest.fixture
def settings() -> Settings:
    return Settings(chunk_max_words=120, chunk_overlap_words=20)


@pytest.fixture
def client(settings):
    app = create_app()
    store = InMemoryStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
