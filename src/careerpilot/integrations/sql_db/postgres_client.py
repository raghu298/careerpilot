"""SQL gateway for the Analytics Agent.

Three stacked read-only guardrails, in order of what actually matters:
1. the LLM's own instruction-following (it is told to only ever SELECT) — unenforced;
2. a deterministic regex pre-filter here that rejects anything but a single SELECT
   before it is ever sent to the database;
3. in production, a dedicated read-only Postgres role — the only unbypassable layer,
   enforced by Postgres itself, not by application code.

The real adapter introspects the live schema so the model never guesses table or
column names. The fake implements just enough of a read-only SELECT engine to drive
the tests offline.
"""

from __future__ import annotations

import re
from typing import Protocol

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|merge)\b",
    re.IGNORECASE,
)


class ReadOnlyViolation(Exception):
    """Raised when a query is not a single read-only SELECT."""


def validate_readonly(query: str) -> str:
    stripped = query.strip().rstrip(";").strip()
    if ";" in stripped:
        raise ReadOnlyViolation("Multiple statements are not allowed")
    if not stripped.lower().startswith(("select", "with")):
        raise ReadOnlyViolation("Only SELECT queries are allowed")
    if _WRITE_KEYWORDS.search(stripped):
        raise ReadOnlyViolation("Write/DDL keywords are not allowed")
    return stripped


class SQLClient(Protocol):
    async def schema_description(self) -> str: ...
    async def run_query(self, query: str) -> list[dict]: ...


class PostgresSQLClient:
    """Real adapter. The injected pool should be connected as a read-only role."""

    def __init__(self, pool, *, role: str) -> None:
        self._pool = pool
        self._role = role

    async def schema_description(self) -> str:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position"
            )
        lines: dict[str, list[str]] = {}
        for r in rows:
            lines.setdefault(r["table_name"], []).append(f"{r['column_name']} {r['data_type']}")
        return "\n".join(f"{t}({', '.join(cols)})" for t, cols in lines.items())

    async def run_query(self, query: str) -> list[dict]:
        safe = validate_readonly(query)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(safe)
        return [dict(r) for r in rows]
