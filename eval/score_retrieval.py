"""Score a retriever against the 10-K eval set.

Input: a JSONL file with one line per question:
    {"id": "aapl-01", "retrieved": [{"doc_id": "AAPL-10K-FY2026", "text": "..."}, ...]}
ranked best first. Chunks can be any size; the scorer only looks at their text.

An evidence span counts as covered by a chunk when the normalized span is a substring
of the normalized chunk, or when at least --overlap of the span's tokens appear in the
chunk (this tolerates small differences between HTML-to-text converters).

Metrics (unanswerable questions are left out and reported separately):
  evidence_recall@k  share of evidence spans covered somewhere in the top k
  full_hit@k         share of questions with every span covered in the top k
  doc_hit@k          share of questions whose top k includes a chunk from a correct doc
  mrr                mean reciprocal rank of the first chunk covering any span

Usage:
    python eval/score_retrieval.py results.jsonl [--k 1 3 5 10] [--json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from textutil import normalize, tokens  # noqa: E402

QUESTIONS = Path(__file__).parent / "10k_v1" / "questions.jsonl"


def covers(span: str, chunk: str, overlap: float) -> bool:
    ns, nc = normalize(span), normalize(chunk)
    if ns in nc:
        return True
    st = tokens(span)
    if not st:
        return False
    have = Counter(tokens(chunk))
    need = Counter(st)
    hit = sum(min(c, have[t]) for t, c in need.items())
    return hit / len(st) >= overlap


def score(questions, results, ks, overlap):
    by_id = {r["id"]: r.get("retrieved", []) for r in results}
    agg = defaultdict(list)
    per_type = defaultdict(lambda: defaultdict(list))
    rows = []
    unanswerable = []
    for q in questions:
        retrieved = by_id.get(q["id"], [])
        if not q["evidence"]:
            unanswerable.append(q["id"])
            continue
        # rank (1-based) at which each span is first covered
        first = []
        for span in q["evidence"]:
            r = next((i + 1 for i, c in enumerate(retrieved) if covers(span, c.get("text", ""), overlap)), None)
            first.append(r)
        doc_rank = next((i + 1 for i, c in enumerate(retrieved) if c.get("doc_id") in q["doc_ids"]), None)
        any_rank = min((r for r in first if r), default=None)
        row = {"id": q["id"], "mrr": 1 / any_rank if any_rank else 0.0}
        for k in ks:
            row[f"evidence_recall@{k}"] = sum(1 for r in first if r and r <= k) / len(first)
            row[f"full_hit@{k}"] = float(all(r and r <= k for r in first))
            row[f"doc_hit@{k}"] = float(bool(doc_rank and doc_rank <= k))
        rows.append(row)
        for m, v in row.items():
            if m != "id":
                agg[m].append(v)
                per_type[q["question_type"]][m].append(v)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "n_scored": len(rows),
        "missing_results": [q["id"] for q in questions if q["id"] not in by_id],
        "unanswerable_skipped": unanswerable,
        "overall": {m: mean(v) for m, v in agg.items()},
        "by_question_type": {t: {m: mean(v) for m, v in ms.items()} for t, ms in per_type.items()},
        "per_question": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--overlap", type=float, default=0.9)
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    a = ap.parse_args()
    qs = [json.loads(l) for l in Path(a.questions).read_text().splitlines() if l.strip()]
    rs = [json.loads(l) for l in Path(a.results).read_text().splitlines() if l.strip()]
    rep = score(qs, rs, a.k, a.overlap)
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    print(f"scored {rep['n_scored']} questions ({len(rep['unanswerable_skipped'])} unanswerable skipped)")
    if rep["missing_results"]:
        print(f"no results for: {', '.join(rep['missing_results'])}")
    for m, v in rep["overall"].items():
        print(f"  {m:22s} {v}")
    print("by question type (evidence_recall@5):")
    for t, ms in sorted(rep["by_question_type"].items()):
        print(f"  {t:14s} {ms.get('evidence_recall@5')}")


if __name__ == "__main__":
    main()
