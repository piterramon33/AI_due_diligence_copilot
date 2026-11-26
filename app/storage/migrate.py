"""Apply migrations/*.sql in filename order, once each. Usage: DATABASE_URL=... python -m app.storage.migrate"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from app.config import get_settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def migrate(dsn: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        for path in sorted(directory.glob("*.sql")):
            if path.name in done:
                continue
            with conn.transaction():
                conn.execute(path.read_text())
                conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


if __name__ == "__main__":
    dsn = get_settings().database_url
    if not dsn:
        sys.exit("Set DATABASE_URL first.")
    for name in migrate(dsn):
        print(f"applied {name}")
