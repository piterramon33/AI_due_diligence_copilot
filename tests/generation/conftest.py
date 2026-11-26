import pytest

from app.generation import SimpleChunk

A = "doc_aaaaaaaaaaaaaaaa:c0001"
B = "doc_aaaaaaaaaaaaaaaa:c0002"
C = "doc_bbbbbbbbbbbbbbbb:c0010"


@pytest.fixture
def chunks():
    return [
        SimpleChunk(A, "Revenue grew 18.5% to $1,240 million in fiscal 2024.",
                    {"title": "Acme 10-K 2024", "doc_type": "filing", "page_start": 41, "page_end": 42}),
        SimpleChunk(B, "Our top three customers accounted for 46% of revenue.",
                    {"title": "Acme 10-K 2024", "doc_type": "filing", "page_start": 12}),
        SimpleChunk(C, "The European market for industrial sensors is expected to grow at 9% annually.",
                    {"title": "Sensor Market Report", "doc_type": "market_report"}),
    ]
