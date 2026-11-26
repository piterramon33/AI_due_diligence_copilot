"""End-to-end check of eval/run_retrieval.py on synthetic filings (no SEC download, no models).

The synthetic filings hold every evidence span, so this tests the plumbing
(ingest -> embed -> retrieve -> score, doc id mapping), not retrieval quality.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "eval" / "10k_v1" / "questions.jsonl"


def _write_fake_filings(out: Path) -> None:
    spans: dict[str, list[str]] = {}
    for line in QUESTIONS.read_text().splitlines():
        q = json.loads(line)
        for d in q["doc_ids"]:
            spans.setdefault(d, []).extend(q["evidence"])
    for doc_id, ss in spans.items():
        body = "".join(
            f"<h2>Item {i % 9 + 1}. Section {i}</h2><p>{html.escape(s)} Other text.</p>"
            for i, s in enumerate(ss)
        )
        (out / f"{doc_id}.htm").write_text(f"<html><body><p>FORM 10-K</p>{body}</body></html>")


@pytest.mark.postgres
def test_runner_end_to_end(tmp_path):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    pytest.importorskip("pgvector")
    _write_fake_filings(tmp_path)
    schema = f"test_eval_{uuid.uuid4().hex[:8]}"
    name = f"pytest-{schema}"
    cmd = [
        sys.executable, str(REPO / "eval" / "run_retrieval.py"),
        "--database-url", url, "--schema", schema, "--docs-dir", str(tmp_path),
        "--embedder", "hashing", "--reranker", "overlap", "--filter", "doc", "--name", name,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        report_path = REPO / "eval" / "runs" / f"{name}.report.json"
        report = json.loads(report_path.read_text())
        assert report["chunking_ceiling"]["findable"] == 1.0
        assert report["overall"]["doc_hit@10"] == 1.0  # filter=doc, so every hit is a right filing
        assert report["run"]["standins"] == ["embedder", "reranker"]
        results = [json.loads(l) for l in (REPO / "eval" / "runs" / f"{name}.jsonl").read_text().splitlines()]
        corpus = json.loads((REPO / "eval" / "10k_v1" / "corpus.json").read_text())["documents"]
        assert {h["doc_id"] for r in results for h in r["retrieved"]} <= {d["doc_id"] for d in corpus}
    finally:
        for f in (REPO / "eval" / "runs").glob(f"{name}*"):
            f.unlink()
        import psycopg

        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
