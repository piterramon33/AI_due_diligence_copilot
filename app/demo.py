"""Preload a real public filing so a fresh install (or the hosted demo) has something to ask about.

The bundled file is Apple Inc.'s Form 10-K for the fiscal year ended September 26, 2026, downloaded
unchanged from SEC EDGAR (public record) and gzipped. Turn preloading off with DDC_DEMO_PRELOAD=0.
"""

from __future__ import annotations

import gzip
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from app.ingestion import ingest_bytes
from app.models import DocType
from app.storage import ChunkStore

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/piterramon33/AI_due_diligence_copilot"
DEMO_DIR = Path(__file__).resolve().parent / "demo_data"


@dataclass(frozen=True)
class DemoFiling:
    path: Path
    filename: str
    company: str
    title: str
    source_url: str


DEMO_FILINGS = [
    DemoFiling(
        path=DEMO_DIR / "aapl-20260926.htm.gz",
        filename="aapl-20260926.htm",
        company="Apple Inc.",
        title="Apple Inc. Form 10-K, fiscal 2026",
        source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019326000026/aapl-20260926.htm",
    ),
]

# Shown as one-click examples in the UI. Phrased to match how the 10-K words things.
SAMPLE_QUESTIONS = [
    "What were total net sales and how did they change from last year?",
    "How much did Apple spend on research and development?",
    "What are the main supply chain risks?",
    "How dependent is Apple on manufacturing outside the U.S.?",
    "How much did the Company repurchase of its common stock during 2026?",
    "What share of revenue comes from Greater China?",
]


def public_demo() -> bool:
    """DDC_PUBLIC_DEMO=1 on a shared hosted instance: uploads are capped and the sample can't be removed."""
    return os.environ.get("DDC_PUBLIC_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}


def max_uploads() -> int:
    """Uploaded (non-sample) documents kept in a public demo; the oldest are dropped first."""
    return int(os.environ.get("DDC_MAX_UPLOADS", "5"))


def enforce_upload_cap(store: ChunkStore) -> None:
    if not public_demo():
        return
    uploads = [d for d in store.list_documents() if not d.metadata.get("demo")]  # newest first
    for doc in uploads[max_uploads():]:
        store.delete(doc.document_id)


def preload_enabled() -> bool:
    return os.environ.get("DDC_DEMO_PRELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}


def preload(store: ChunkStore) -> list[str]:
    """Ingest the bundled filings that are not already in the store. Returns the new document ids."""
    added = []
    known = {d.document_id for d in store.list_documents()}
    for filing in DEMO_FILINGS:
        if not filing.path.exists():
            log.warning("Demo filing missing: %s", filing.path)
            continue
        data = gzip.decompress(filing.path.read_bytes())
        document, chunks = ingest_bytes(data, filing.filename, DocType.filing, filing.company)
        if document.document_id in known:
            continue
        document.title = filing.title
        document.metadata["demo"] = True
        document.metadata["source_url"] = filing.source_url
        for chunk in chunks:
            chunk.metadata["title"] = filing.title
        store.save(document, chunks)
        added.append(document.document_id)
        log.info("Preloaded demo filing %s (%d chunks)", filing.filename, len(chunks))
    return added
