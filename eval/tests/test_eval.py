import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from score_retrieval import QUESTIONS, covers, score  # noqa: E402
from textutil import html_to_text, normalize  # noqa: E402

QS = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
CORPUS = {d["doc_id"] for d in json.loads((HERE / "10k_v1" / "corpus.json").read_text())["documents"]}


class TestDataset(unittest.TestCase):
    def test_schema(self):
        ids = set()
        for q in QS:
            self.assertNotIn(q["id"], ids)
            ids.add(q["id"])
            for f in ("question", "expected_answer", "evidence", "doc_ids", "status", "question_type"):
                self.assertIn(f, q, q["id"])
            self.assertIn(q["status"], {"verified", "derived", "guess"})
            self.assertTrue(set(q["doc_ids"]) <= CORPUS, q["id"])
            if q["question_type"] != "unanswerable":
                self.assertTrue(q["evidence"], q["id"])


class TestScorer(unittest.TestCase):
    def test_covers_whitespace_and_case(self):
        self.assertTrue(covers("iPhone $ 254,098 21 %", "foo\tIPHONE\t$\t254,098 \t21\t% bar", 0.9))
        self.assertFalse(covers("iPhone $ 254,098", "Mac $ 37,929", 0.9))

    def test_perfect_and_empty(self):
        perfect = [{"id": q["id"], "retrieved": [{"doc_id": q["doc_ids"][0], "text": " ".join(q["evidence"])}]} for q in QS]
        rep = score(QS, perfect, [1], 0.9)
        self.assertEqual(rep["overall"]["full_hit@1"], 1.0)
        self.assertEqual(rep["overall"]["mrr"], 1.0)
        empty = score(QS, [{"id": q["id"], "retrieved": []} for q in QS], [1], 0.9)
        self.assertEqual(empty["overall"]["evidence_recall@1"], 0.0)

    def test_html_to_text_real_edgar_markup(self):
        # Rows cut from Apple's FY2026 10-K table markup (style attributes removed).
        html = (Path(__file__).parent / "fixtures" / "aapl_net_sales_rows.html").read_text()
        text = html_to_text(html)
        self.assertNotIn("0000320193", text)  # hidden iXBRL header is dropped
        by_id = {q["id"]: q for q in QS}
        for qid in ("aapl-01", "aapl-06"):
            for span in by_id[qid]["evidence"]:
                self.assertIn(normalize(span), normalize(text), span)
        self.assertIn(normalize(by_id["aapl-02"]["evidence"][0]), normalize(text))


if __name__ == "__main__":
    unittest.main()
