"""Run the 10-K eval questions through the real ingestion + hybrid retriever and score them.

Steps:
  1. ingest each filing in corpus.json with app.ingestion (chunking as in production)
     into Postgres, in its own schema (default `eval_10k_v1`) so app data is untouched
  2. fill embeddings with app.retrieval.embed_pending_chunks
  3. ask every question through app.retrieval.HybridRetriever
  4. write results JSONL in score_retrieval.py's format and score it

Filings are read from eval/10k_v1/docs/<doc_id>.htm (raw EDGAR HTML, best: keeps
tables) or <doc_id>.txt (what fetch_corpus.py saves). `--fetch` downloads the HTML.

Usage:
    DATABASE_URL=postgresql://... python eval/run_retrieval.py                 # real bge models
    DATABASE_URL=... python eval/run_retrieval.py --mode keyword --reranker none   # full-text only, no models
    DATABASE_URL=... python eval/run_retrieval.py --embedder hashing --reranker overlap   # offline stand-ins
    SEC_USER_AGENT="Name you@example.com" DATABASE_URL=... python eval/run_retrieval.py --fetch

Needs: pip install -e ".[retrieval]" (sentence-transformers only for the bge models),
and Postgres with pgvector, e.g. `docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(EVAL))

from score_retrieval import QUESTIONS, covers, score  # noqa: E402

CORPUS = EVAL / "10k_v1" / "corpus.json"
DOCS = EVAL / "10k_v1" / "docs"
RUNS = EVAL / "runs"


def find_doc_file(docs_dir: Path, doc_id: str) -> Path | None:
    for ext in (".htm", ".html", ".txt"):
        p = docs_dir / f"{doc_id}{ext}"
        if p.exists():
            return p
    return None


def fetch_html(docs: list[dict], docs_dir: Path) -> None:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("--fetch needs SEC_USER_AGENT='Name email@domain' (SEC fair-access policy).")
    docs_dir.mkdir(parents=True, exist_ok=True)
    for d in docs:
        target = docs_dir / f"{d['doc_id']}.htm"
        if target.exists():
            continue
        req = urllib.request.Request(d["url"], headers={"User-Agent": ua, "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=60) as r:
            target.write_bytes(r.read())
        print(f"fetched {target.name} ({target.stat().st_size:,} bytes)")
        time.sleep(0.5)


def scoped_dsn(dsn: str, schema: str) -> str:
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    opts = conninfo_to_dict(dsn).get("options") or ""
    return make_conninfo(dsn, options=f"{opts} -c search_path={schema},public".strip())


def prepare_schema(dsn: str, schema: str, reset: bool) -> str:
    import psycopg

    from app.storage.migrate import migrate

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # in public, shared by schemas
        if reset:
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    sdsn = scoped_dsn(dsn, schema)
    migrate(sdsn)
    return sdsn


def ingest(sdsn: str, docs: list[dict], docs_dir: Path) -> dict[str, str]:
    """Ingest each filing once per chunk setting. Returns {app document_id: eval doc_id}."""
    from app.config import get_settings
    from app.ingestion.pipeline import ingest_bytes
    from app.storage.postgres import PostgresStore

    settings = get_settings()
    chunking = {"max_words": settings.chunk_max_words, "overlap_words": settings.chunk_overlap_words}
    store = PostgresStore(sdsn)
    id_map = {}
    for d in docs:
        path = find_doc_file(docs_dir, d["doc_id"])
        if path is None:
            sys.exit(
                f"missing {docs_dir}/{d['doc_id']}.htm (or .txt). Run with --fetch, "
                "or save the filing from its EDGAR URL in corpus.json."
            )
        data = path.read_bytes()
        document, chunks = ingest_bytes(data, path.name, doc_type="filing", company=d["company"])
        id_map[document.document_id] = d["doc_id"]
        old = store.get_document(document.document_id)
        if old and old.metadata.get("eval_chunking") == chunking:
            print(f"{d['doc_id']}: already ingested ({old.metadata.get('chunk_count')} chunks)")
            continue
        document.metadata["eval_chunking"] = chunking
        document.metadata["eval_doc_id"] = d["doc_id"]
        store.save(document, chunks)
        n_tables = sum(c.kind == "table" for c in chunks)
        print(f"{d['doc_id']}: ingested {path.name} -> {len(chunks)} chunks ({n_tables} tables)")
    return id_map


def chunking_ceiling(conn, questions: list[dict], eval_to_app: dict[str, str]) -> dict:
    """Which evidence spans exist in at least one stored chunk.

    A span no chunk covers can never be retrieved, whatever the search does, so
    this separates chunking/parsing losses from ranking losses.
    """
    by_doc: dict[str, list[str]] = {}
    for doc_id, content in conn.execute(
        "SELECT document_id, content FROM chunks WHERE document_id = ANY(%s)", [list(eval_to_app.values())]
    ):
        by_doc.setdefault(doc_id, []).append(content)
    total, missing = 0, []
    for q in questions:
        texts = [t for d in q["doc_ids"] for t in by_doc.get(eval_to_app.get(d), [])]
        for span in q["evidence"]:
            total += 1
            if not any(covers(span, t, 0.9) for t in texts):
                missing.append({"id": q["id"], "span": span})
    return {
        "spans": total,
        "findable": round((total - len(missing)) / total, 4) if total else None,
        "unfindable": missing,
    }


def build_components(args):
    if args.embedder == "bge":
        from app.retrieval import SentenceTransformerEmbedder

        embedder, model_name = SentenceTransformerEmbedder(args.embedding_model), args.embedding_model
    else:
        from standins import HashingEmbedder

        embedder, model_name = HashingEmbedder(), "standin-hashing-384"
    if args.reranker == "bge":
        from app.retrieval import CrossEncoderReranker

        reranker = CrossEncoderReranker(args.rerank_model)
    elif args.reranker == "overlap":
        from standins import OverlapReranker

        reranker = OverlapReranker()
    else:
        reranker = None
    return embedder, model_name, reranker


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--schema", default="eval_10k_v1")
    ap.add_argument("--reset", action="store_true", help="drop the eval schema and re-ingest")
    ap.add_argument("--docs-dir", type=Path, default=DOCS)
    ap.add_argument("--fetch", action="store_true", help="download missing filings from EDGAR first")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    ap.add_argument("--embedder", choices=["bge", "hashing"], default="bge")
    ap.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--reranker", choices=["bge", "overlap", "none"], default="bge")
    ap.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    ap.add_argument("--rerank-top-n", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=50, help="per first-stage retriever")
    ap.add_argument("--top-k", type=int, default=10, help="chunks kept per question")
    ap.add_argument(
        "--filter", choices=["none", "doc"], default="none",
        help="'doc' restricts search to the question's filings (as if the user picked them)",
    )
    ap.add_argument("--name", help="run name for files under eval/runs/ (default: from options)")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    args = ap.parse_args()
    if not args.database_url:
        sys.exit("Set DATABASE_URL (Postgres with pgvector) or pass --database-url.")
    if args.mode == "keyword":
        args.embedder = "hashing"  # never queried (vector_k=0); avoids loading a model

    import psycopg

    from app.retrieval import HybridRetriever, PgSearchStore, RetrievalConfig, SearchFilters
    from app.retrieval.indexer import embed_pending_chunks

    docs = json.loads(CORPUS.read_text())["documents"]
    if args.fetch:
        fetch_html(docs, args.docs_dir)
    sdsn = prepare_schema(args.database_url, args.schema, args.reset)
    id_map = ingest(sdsn, docs, args.docs_dir)
    eval_to_app = {v: k for k, v in id_map.items()}

    embedder, model_name, reranker = build_components(args)
    conn = psycopg.connect(sdsn)
    if args.mode != "keyword":
        t = time.time()
        n = embed_pending_chunks(conn, embedder, model_name=model_name)
        if n:
            print(f"embedded {n} chunks with {model_name} in {time.time() - t:.0f}s")

    cfg = RetrievalConfig(
        vector_k=0 if args.mode == "keyword" else args.candidates,
        keyword_k=0 if args.mode == "vector" else args.candidates,
        rerank_top_n=args.rerank_top_n,
        top_k=args.top_k,
    )
    retriever = HybridRetriever(PgSearchStore(conn), embedder, reranker, cfg)

    questions = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()]
    results, t = [], time.time()
    for q in questions:
        filters = None
        if args.filter == "doc":
            filters = SearchFilters(document_ids=[eval_to_app[d] for d in q["doc_ids"]])
        hits = retriever.retrieve(q["question"], filters, top_k=args.top_k)
        results.append({
            "id": q["id"],
            "retrieved": [
                {
                    "doc_id": id_map.get(h.document_id, h.document_id),
                    "chunk_id": h.chunk_id,
                    "section": h.metadata.get("section"),
                    "kind": h.metadata.get("kind"),
                    "score": h.score,
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                    "text": h.text,
                }
                for h in hits
            ],
        })
    elapsed = time.time() - t
    ceiling = chunking_ceiling(conn, questions, eval_to_app)
    conn.close()

    standins = [x for x, on in (("embedder", args.embedder == "hashing" and args.mode != "keyword"),
                                ("reranker", args.reranker == "overlap")) if on]
    name = args.name or "-".join(
        [args.mode, args.embedder if args.mode != "keyword" else "nofx", args.reranker, f"filter_{args.filter}"]
    )
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"{name}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in results))
    report = score(questions, results, args.k, overlap=0.9)
    report["run"] = {
        "name": name,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": args.mode,
        "embedder": model_name if args.mode != "keyword" else None,
        "reranker": {"bge": args.rerank_model, "overlap": "standin-overlap", "none": None}[args.reranker],
        "standins": standins,
        "filter": args.filter,
        "config": cfg.__dict__,
        "seconds_per_question": round(elapsed / max(len(questions), 1), 3),
    }
    report["chunking_ceiling"] = ceiling
    (RUNS / f"{name}.report.json").write_text(json.dumps(report, indent=2))

    print(f"\n{name}: {report['n_scored']} questions scored, {elapsed / len(questions):.2f}s/question")
    if standins:
        print(f"  NOTE: stand-in {' and '.join(standins)} used; these are not the real model scores")
    for m, v in report["overall"].items():
        print(f"  {m:22s} {v}")
    print(f"  {'spans in any chunk':22s} {ceiling['findable']} "
          f"({len(ceiling['unfindable'])} of {ceiling['spans']} lost to parsing/chunking)")
    print("by question type (evidence_recall@5):")
    for qt, ms in sorted(report["by_question_type"].items()):
        print(f"  {qt:14s} {ms.get('evidence_recall@5')}")
    print(f"results: {out}\nreport:  {RUNS / (name + '.report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
