"""Download the 10-K filings listed in corpus.json from SEC EDGAR and save them as plain text.

SEC requires a descriptive User-Agent with a contact address:
    SEC_USER_AGENT="YourName your@email.com" python eval/fetch_corpus.py
Then check that every evidence span is still found in the text:
    python eval/fetch_corpus.py --verify
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from textutil import html_to_text, normalize  # noqa: E402

ROOT = Path(__file__).parent / "10k_v1"


def fetch(url: str, ua: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def verify() -> int:
    docs = {}
    for f in (ROOT / "docs").glob("*.txt"):
        docs[f.stem] = normalize(f.read_text())
    missing = 0
    for line in (ROOT / "questions.jsonl").read_text().splitlines():
        q = json.loads(line)
        for span in q["evidence"]:
            if not any(normalize(span) in docs.get(d, "") for d in q["doc_ids"]):
                missing += 1
                print(f"MISSING {q['id']}: {span[:80]}")
    print("all evidence spans found" if not missing else f"{missing} spans missing")
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only check evidence spans against saved text")
    args = ap.parse_args()
    if args.verify:
        return verify()
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("Set SEC_USER_AGENT to 'Name email@domain' (SEC fair-access policy).")
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    for d in json.loads((ROOT / "corpus.json").read_text())["documents"]:
        target = out / f"{d['doc_id']}.txt"
        if target.exists():
            print(f"skip {d['doc_id']} (exists)")
            continue
        html = fetch(d["url"], ua)
        target.write_text(html_to_text(html))
        print(f"saved {target} ({target.stat().st_size:,} bytes)")
        time.sleep(0.5)  # stay well under SEC's 10 req/s limit
    return verify()


if __name__ == "__main__":
    sys.exit(main())
