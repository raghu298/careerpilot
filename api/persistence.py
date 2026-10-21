"""Durable persistence for demo sessions — so a server restart / Render cold-start no
longer wipes a visitor's session, application tracker, or conversation memory.

A single SQLite file (``CP_DB_PATH``, default ``careerpilot_data.db``; set to
``:memory:`` to disable durability). SQLite here is a real database, same as the
application tracker. The ``SessionStore`` rehydrates a session from this store on the
first request after a restart.

Production note: on a host with a persistent disk (or a mounted volume) this survives
restarts as-is. Render's *free* web tier has an ephemeral filesystem, so to make the
public demo durable there, point ``CP_DB_PATH`` at a mounted disk or run against an
external Postgres (same table shapes) — the `SessionStore` depends only on this class's
methods, so swapping the backend touches nothing else.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    persona_id  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_seen   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS applications (
    token TEXT NOT NULL,
    seq   INTEGER NOT NULL,
    data  TEXT NOT NULL,
    PRIMARY KEY (token, seq)
);
CREATE TABLE IF NOT EXISTS memory (
    token   TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (token, seq)
);
"""


def default_db_path() -> str:
    return os.environ.get("CP_DB_PATH", "careerpilot_data.db")


class PersistenceStore:
    def __init__(self, path: str | None = None) -> None:
        self._path = path or default_db_path()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._lock = threading.Lock()

    # ---- sessions ----
    def save_session(self, token: str, persona_id: str, created_at: float, last_seen: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (token, persona_id, created_at, last_seen) "
                "VALUES (?, ?, ?, ?)",
                (token, persona_id, created_at, last_seen),
            )
            self._conn.commit()

    def touch_session(self, token: str, last_seen: float) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET last_seen = ? WHERE token = ?", (last_seen, token))
            self._conn.commit()

    def load_session(self, token: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT token, persona_id, created_at, last_seen FROM sessions WHERE token = ?",
                (token,),
            ).fetchone()
        return dict(row) if row else None

    def delete_session(self, token: str) -> None:
        with self._lock:
            for t in ("sessions", "applications", "memory"):
                self._conn.execute(f"DELETE FROM {t} WHERE token = ?", (token,))
            self._conn.commit()

    # ---- applications ----
    def save_applications(self, token: str, apps: list[dict]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM applications WHERE token = ?", (token,))
            self._conn.executemany(
                "INSERT INTO applications (token, seq, data) VALUES (?, ?, ?)",
                [(token, i, json.dumps(a)) for i, a in enumerate(apps)],
            )
            self._conn.commit()

    def load_applications(self, token: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data FROM applications WHERE token = ? ORDER BY seq", (token,)
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    # ---- memory ----
    def save_memory(self, token: str, messages: list[dict]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE token = ?", (token,))
            self._conn.executemany(
                "INSERT INTO memory (token, seq, role, content) VALUES (?, ?, ?, ?)",
                [(token, i, m["role"], m["content"]) for i, m in enumerate(messages)],
            )
            self._conn.commit()

    def load_memory(self, token: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM memory WHERE token = ? ORDER BY seq", (token,)
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ---- maintenance ----
    def purge_expired(self, ttl_seconds: float) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            tokens = [
                r["token"]
                for r in self._conn.execute(
                    "SELECT token FROM sessions WHERE last_seen < ?", (cutoff,)
                ).fetchall()
            ]
            for token in tokens:
                for t in ("sessions", "applications", "memory"):
                    self._conn.execute(f"DELETE FROM {t} WHERE token = ?", (token,))
            self._conn.commit()
        return len(tokens)
