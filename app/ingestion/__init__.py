"""Document ingestion: parse files into ordered blocks, assign sections, chunk with citation metadata."""

from app.ingestion.pipeline import IngestError, ingest_bytes

__all__ = ["IngestError", "ingest_bytes"]
