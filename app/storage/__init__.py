"""Document/chunk persistence. Postgres when DATABASE_URL is set, in-memory otherwise."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.storage.base import ChunkStore
from app.storage.memory import InMemoryStore


@lru_cache
def get_store() -> ChunkStore:
    settings = get_settings()
    if settings.database_url:
        from app.storage.postgres import PostgresStore

        return PostgresStore(settings.database_url)
    return InMemoryStore()


__all__ = ["ChunkStore", "InMemoryStore", "get_store"]
