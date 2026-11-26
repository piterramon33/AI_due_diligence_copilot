"""Score generated answers against the 10-K eval set (numbers and abstention only).

Input: the same JSONL as score_retrieval.py, with an "answer" string per line:
    {"id": "aapl-01", "answer": "Apple's net sales were $478.8 billion ...", "retrieved": [...]}

Checks:
  numeric_accuracy   share of questions with answer_value where some number in the answer
                     matches it within tolerance (scale words like "billion" are honored)
  abstention_rate    share of unanswerable questions whose answer says the figure is not disclosed
  false_abstention   share of answerable questions whose answer abstains anyway
                     (false_premise questions are left out of this one)

This is deliberately simple. It does not judge prose quality or citations.

Usage:
    python eval/score_answers.py results.jsonl [--json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from textutil import normalize  # noqa: E402

QUESTIONS = Path(__file__).parent / "10k_v1" / "questions.jsonl"

UNIT_FACTOR = {"USD millions": 1e6, "USD billions": 1e9, "ratio": 1.0, "employees": 1.0}
SCALE = {"thousand": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
_NUM = re.compile(r"(?<![\w.])\(?\$?\s?(-?\d[\d,]*(?:\.\d+)?)\s?(%|percent|thousand|million|billion|trillion|bn|m|b)?(?![a-z])", re.I)
ABSTAIN = re.compile(
    r"not (separately |specifically )?(disclosed|reported|provided|available|stated|broken out)"
    r"|does not (disclose|report|provide|break out)|doesn't (disclose|report|provide)"
    r"|no (such )?(figure|data|disclosure|information)|cannot (be )?(determined|found|answered)"
    r"|isn't (disclosed|reported)|unable to (find|determine)",
    re.I,
)


def numbers(text: str) -> list[tuple[float, bool]]:
    """Return (value, had_scale_word) pairs. Percentages become ratios."""
    out = []
    for m in _NUM.finditer(text):
        raw, scale = m.group(1).replace(",", ""), (m.group(2) or "").lower()
        try:
            v = float(raw)
        except ValueError:
            continue
        if scale in ("%", "percent"):
            out.append((v / 100, True))
        elif scale in SCALE:
            out.append((v * SCALE[scale], True))
        else:
            out.append((v, False))
    return out


def number_matches(answer: str, value: float, unit: str, tol: float) -> bool:
    factor = UNIT_FACTOR.get(unit, 1.0)
    target = value * factor
    tol = max(tol, 1e-9)
    for v, scaled in numbers(answer):
        cands = [v] if scaled else [v, v * factor]  # bare numbers may already be in the question's unit
        if any(abs(c - target) <= tol * abs(target) + 1e-9 for c in cands):
            return True
    return False


def abstains(answer: str) -> bool:
    return bool(ABSTAIN.search(normalize(answer)))


def score(questions, results):
    by_id = {r["id"]: r for r in results}
    num_hits, num_total, abst, abst_total, false_abst, answerable = 0, 0, 0, 0, 0, 0
    rows = []
    for q in questions:
        r = by_id.get(q["id"])
        if not r or "answer" not in r:
            continue
        a = r["answer"]
        row = {"id": q["id"]}
        if q["question_type"] == "unanswerable":
            abst_total += 1
            row["abstained"] = abstains(a)
            abst += row["abstained"]
        elif q["question_type"] != "false_premise":  # correcting a false premise may read like abstaining
            answerable += 1
            if abstains(a):
                false_abst += 1
                row["false_abstention"] = True
            if "answer_value" in q:
                num_total += 1
                row["numeric_match"] = number_matches(a, q["answer_value"], q.get("unit", ""), q.get("tolerance", 0.0))
                num_hits += row["numeric_match"]
        rows.append(row)
    ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "numeric_accuracy": ratio(num_hits, num_total),
        "numeric_n": num_total,
        "abstention_rate": ratio(abst, abst_total),
        "abstention_n": abst_total,
        "false_abstention": ratio(false_abst, answerable),
        "per_question": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    qs = [json.loads(l) for l in Path(a.questions).read_text().splitlines() if l.strip()]
    rs = [json.loads(l) for l in Path(a.results).read_text().splitlines() if l.strip()]
    rep = score(qs, rs)
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    for k in ("numeric_accuracy", "numeric_n", "abstention_rate", "abstention_n", "false_abstention"):
        print(f"  {k:18s} {rep[k]}")
    misses = [r["id"] for r in rep["per_question"] if r.get("numeric_match") is False]
    if misses:
        print("numeric misses:", ", ".join(misses))


if __name__ == "__main__":
    main()
