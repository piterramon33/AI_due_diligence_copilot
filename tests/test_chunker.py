import pytest

from app.ingestion.chunker import chunk_blocks
from app.ingestion.parsers.base import Block


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunks_never_cross_sections():
    blocks = [
        Block("heading", "Item 1. Business", 1, level=2),
        Block("text", words(30, "a"), 1),
        Block("heading", "Item 1A. Risk Factors", 2, level=2),
        Block("text", words(30, "b"), 2),
    ]
    chunks = chunk_blocks(blocks, max_words=100, overlap_words=10)
    assert [c.section_path for c in chunks] == [["Item 1. Business"], ["Item 1A. Risk Factors"]]
    assert "b0" not in chunks[0].text and "a29" not in chunks[1].text


def test_long_text_splits_with_overlap_and_exact_pages():
    blocks = [Block("heading", "Item 7. MD&A", 1, level=2), Block("text", words(80), 1), Block("text", words(80, "x"), 2)]
    chunks = chunk_blocks(blocks, max_words=60, overlap_words=10)
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.text.split()) <= 60
    # overlap: last 10 words of chunk 0 start chunk 1
    assert chunks[1].text.split()[:10] == chunks[0].text.split()[-10:]
    assert chunks[0].page_start == chunks[0].page_end == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)
    # every source word survives
    joined = " ".join(c.text for c in chunks).split()
    assert set(words(80).split()) <= set(joined) and set(words(80, "x").split()) <= set(joined)


def test_paragraph_kept_whole_when_it_fits_next_chunk():
    blocks = [Block("text", words(40, "a"), 1), Block("text", words(30, "b"), 1)]
    chunks = chunk_blocks(blocks, max_words=50, overlap_words=5)
    assert len(chunks) == 2
    assert "b0" not in chunks[0].text  # second paragraph not split across chunks
    assert words(30, "b") in chunks[1].text


def test_no_overlap_only_chunk_at_section_end():
    # 90 words = 50 + (10 overlap + 40); the 10 carried words alone must not become a third chunk.
    blocks = [Block("text", words(90), 1), Block("heading", "Next", 1, level=3), Block("text", "tail", 1)]
    chunks = chunk_blocks(blocks, max_words=50, overlap_words=10)
    assert [len(c.text.split()) for c in chunks] == [50, 50, 1]


def test_big_table_split_by_rows_repeats_header():
    header = ["Metric", "2025", "2024"]
    rows = [header] + [[f"Line {i}", "1", "2"] for i in range(40)]
    from app.ingestion.parsers.base import table_to_markdown

    block = Block("table", table_to_markdown(rows), 5, rows=rows)
    chunks = chunk_blocks([block], max_words=20, overlap_words=0)
    assert len(chunks) > 1
    assert all(c.kind == "table" and c.text.startswith("| Metric | 2025 | 2024 |") for c in chunks)
    assert all(c.page_start == 5 for c in chunks)


def test_invalid_size():
    with pytest.raises(ValueError):
        chunk_blocks([], max_words=0)
