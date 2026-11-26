"""Runtime settings, read from environment variables prefixed with DDC_ (plus DATABASE_URL)."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Diligence Copilot"
    # Postgres DSN. When empty, an in-memory store is used (fine for dev and tests).
    database_url: str = ""
    # Guess: 50 MB covers most 10-Ks and decks; raise for scanned exhibits.
    max_upload_mb: int = 50
    # Guess: ~350 words ≈ 450-500 tokens, a common sweet spot for financial text retrieval.
    chunk_max_words: int = 350
    chunk_overlap_words: int = 50

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            database_url=env.get("DATABASE_URL", ""),
            max_upload_mb=int(env.get("DDC_MAX_UPLOAD_MB", cls.model_fields["max_upload_mb"].default)),
            chunk_max_words=int(env.get("DDC_CHUNK_MAX_WORDS", cls.model_fields["chunk_max_words"].default)),
            chunk_overlap_words=int(
                env.get("DDC_CHUNK_OVERLAP_WORDS", cls.model_fields["chunk_overlap_words"].default)
            ),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
