"""An in-memory read-only SQL stand-in for the application tracker (offline demo/tests).

It is not a real SQL engine. It holds a seeded ``applications`` table and answers a
small, fixed set of tracker questions by pattern — enough to exercise the Analytics
Agent's tool-calling loop and the read-only guardrail without a database. Every query
still passes through ``validate_readonly`` so the guardrail is tested on the same path
the real adapter uses.
"""

from __future__ import annotations

from careerpilot.integrations.sql_db.postgres_client import validate_readonly

_SCHEMA = (
    "applications(id int, company text, role text, status text, applied_on date, "
    "last_update date, source text, location text)\n"
    "-- status values: Applied, Screening, Interviewing, Offer, Rejected, Ghosted"
)

# Seeded application history for the demo persona.
_APPS = [
    {"company": "Nimbus Labs", "role": "Senior GenAI Engineer", "status": "Interviewing", "applied_on": "2026-09-28", "last_update": "2026-10-18"},
    {"company": "Flowbit", "role": "Backend Engineer (AI)", "status": "Screening", "applied_on": "2026-10-02", "last_update": "2026-10-09"},
    {"company": "Coreline", "role": "Staff ML Engineer", "status": "Applied", "applied_on": "2026-09-15", "last_update": "2026-09-15"},
    {"company": "Deltastack", "role": "Platform Engineer", "status": "Rejected", "applied_on": "2026-09-10", "last_update": "2026-09-22"},
    {"company": "Ver1", "role": "GenAI Engineer", "status": "Ghosted", "applied_on": "2026-08-30", "last_update": "2026-08-30"},
    {"company": "Brightmint", "role": "Senior Backend Engineer", "status": "Offer", "applied_on": "2026-09-05", "last_update": "2026-10-14"},
]

_STATUS_COUNTS = [
    {"status": "Applied", "count": 1},
    {"status": "Screening", "count": 1},
    {"status": "Interviewing", "count": 1},
    {"status": "Offer", "count": 1},
    {"status": "Rejected", "count": 1},
    {"status": "Ghosted", "count": 1},
]


class FakeSQLClient:
    def __init__(self, *, role: str = "basic") -> None:
        self._role = role

    async def schema_description(self) -> str:
        return _SCHEMA

    async def run_query(self, query: str) -> list[dict]:
        safe = validate_readonly(query).lower()
        if "group by" in safe and "status" in safe:
            return list(_STATUS_COUNTS)
        if "ghost" in safe or "stale" in safe or ("last_update" in safe and "where" in safe):
            return [a for a in _APPS if a["status"] in ("Ghosted", "Applied")]
        if "count(" in safe and "status" not in safe:
            return [{"total_applications": len(_APPS)}]
        # Fallback: the full tracker.
        return list(_APPS)
