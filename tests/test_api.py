def test_upload_list_get_chunks_delete(client, sample_10k_pdf):
    r = client.post(
        "/documents",
        files={"file": ("acme-10k.pdf", sample_10k_pdf, "application/pdf")},
        data={"doc_type": "filing", "company": "Acme"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    doc_id = body["document"]["document_id"]
    assert doc_id.startswith("doc_")
    assert body["table_count"] == 1
    assert "Item 1A. Risk Factors" in body["sections"]
    assert body["warnings"] == []

    assert [d["document_id"] for d in client.get("/documents").json()] == [doc_id]
    assert client.get(f"/documents/{doc_id}").json()["company"] == "Acme"

    chunks = client.get(f"/documents/{doc_id}/chunks").json()
    assert chunks and all({"chunk_id", "page_start", "page_end", "section", "section_path"} <= c.keys() for c in chunks)

    page3 = client.get(f"/documents/{doc_id}/chunks", params={"page": 3}).json()
    assert page3 and all(c["page_start"] <= 3 <= c["page_end"] for c in page3)
    risk = client.get(f"/documents/{doc_id}/chunks", params={"section": "risk factors"}).json()
    assert risk and all(c["section"] == "Item 1A. Risk Factors" for c in risk)

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_reupload_is_idempotent(client, sample_10k_html):
    files = {"file": ("acme.htm", sample_10k_html, "text/html")}
    a = client.post("/documents", files=files).json()
    b = client.post("/documents", files=files).json()
    assert a["document"]["document_id"] == b["document"]["document_id"]
    assert len(client.get("/documents").json()) == 1


def test_rejects_unsupported_and_oversized(client, settings):
    r = client.post("/documents", files={"file": ("deck.pptx", b"PK\x03\x04junk", "application/octet-stream")})
    assert r.status_code == 422
    assert "Unsupported" in r.json()["detail"]

    settings.max_upload_mb = 0
    r = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 413


def test_markdown_report(client):
    md = b"# Market Overview\n\nThe sensor market grows 8% a year.\n\n## Competitors\n\nTwo incumbents hold 60% share.\f# Outlook\n\nStrong."
    r = client.post("/documents", files={"file": ("report.md", md, "text/markdown")}, data={"doc_type": "market_report"})
    assert r.status_code == 201
    doc_id = r.json()["document"]["document_id"]
    chunks = client.get(f"/documents/{doc_id}/chunks").json()
    comp = next(c for c in chunks if "incumbents" in c["text"])
    assert comp["section_path"] == ["Market Overview", "Competitors"]
    outlook = next(c for c in chunks if c["text"] == "Strong.")
    assert outlook["page_start"] == 2 and outlook["section_path"] == ["Outlook"]
