"""A REAL SQL engine for the Analytics Agent's application tracker — SQLite (stdlib).

This is not a pattern-matching stand-in: it is an actual SQL database. The Analytics
agent's generated ``SELECT`` runs through the same read-only guardrail
(:func:`validate_readonly`) and is then executed by SQLite, so the numbers come from
real query execution over real rows the user owns. The ``PostgresSQLClient`` remains the
production adapter when a managed read-only Postgres role is available; SQLite is the
zero-config real engine used by the demo and offline (no DB server, no credentials, no
card).

Writes (loading the user's applications) go through :meth:`replace_applications`, which
is called by the API only — never by the agent, which is read-only.
"""

from __future__ import annotations

import asyncio
import sqlite3

from careerpilot.integrations.sql_db.postgres_client import validate_readonly

_DDL = """
CREATE TABLE IF NOT EXISTS applications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company     TEXT NOT NULL,
    role        TEXT NOT NULL,
    status      TEXT NOT NULL,
    applied_on  TEXT,
    last_update TEXT,
    source      TEXT,
    location    TEXT
);
"""

# A small, clearly-labelled starter set so a brand-new session isn't empty. The user
# replaces it with their own via the API ("My applications"); nothing here is hard-coded
# into answers — every answer is a live SQL query over whatever rows the table holds.
_STARTER_ROWS = [
    ("Nimbus Labs", "Senior GenAI Engineer", "Interviewing", "2026-09-28", "2026-10-18", "LinkedIn", "Remote"),
    ("Flowbit", "Backend Engineer (AI)", "Screening", "2026-10-02", "2026-10-09", "Referral", "Bangalore"),
    ("Coreline", "Staff ML Engineer", "Applied", "2026-09-15", "2026-09-15", "Company site", "Remote"),
    ("Deltastack", "Platform Engineer", "Rejected", "2026-09-10", "2026-09-22", "LinkedIn", "Hyderabad"),
    ("Ver1", "GenAI Engineer", "Ghosted", "2026-08-30", "2026-08-30", "Naukri", "Remote"),
    ("Brightmint", "Senior Backend Engineer", "Offer", "2026-09-05", "2026-10-14", "Referral", "Bangalore"),
]

_COLUMNS = ("company", "role", "status", "applied_on", "last_update", "source", "location")


class SqliteSQLClient:
    """Real read-only-for-the-agent SQLite tracker. One in-memory DB per session."""

    def __init__(self, *, role: str = "basic", seed: bool = True) -> None:
        self._role = role
        # check_same_thread=False: the async server may touch it from a worker thread.
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._conn.executescript(_DDL)
        if seed:
            self._insert(_STARTER_ROWS)

    # ---- write path (API only, never the agent) ----------------------------

    def _insert(self, rows) -> None:
        self._conn.executemany(
            f"INSERT INTO applications ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
            rows,
        )
        self._conn.commit()

    def replace_applications(self, apps: list[dict]) -> int:
        """Replace all rows with the user's real applications. Returns the row count."""
        rows = [tuple((a.get(c) or None) for c in _COLUMNS) for a in apps]
        self._conn.execute("DELETE FROM applications")
        self._insert(rows)
        return len(rows)

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]

    def all_rows(self) -> list[dict]:
        """Every application as a dict (for durable persistence)."""
        cols = ", ".join(_COLUMNS)
        rows = self._conn.execute(f"SELECT {cols} FROM applications ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ---- SQLClient protocol (read-only) -------------------------------------

    async def schema_description(self) -> str:
        # Real introspection via PRAGMA — the model never guesses column names.
        cols = self._conn.execute("PRAGMA table_info(applications)").fetchall()
        coldefs = ", ".join(f"{c['name']} {c['type'].lower()}" for c in cols)
        statuses = "Applied, Screening, Interviewing, Offer, Rejected, Ghosted"
        return f"applications({coldefs})\n-- status values: {statuses}"

    async def run_query(self, query: str) -> list[dict]:
        safe = validate_readonly(query)  # raises ReadOnlyViolation on any write/DDL
        async with self._lock:
            cur = self._conn.execute(safe)
            rows = cur.fetchall()
        return [dict(r) for r in rows]
