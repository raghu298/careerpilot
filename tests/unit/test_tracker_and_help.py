"""Tests for the real SQLite application tracker and the orchestrator help intercept."""

from careerpilot.integrations.sql_db.postgres_client import ReadOnlyViolation
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient
from careerpilot.orchestrator.orchestrator import _help_reply


async def test_sqlite_tracker_real_query_over_user_rows():
    db = SqliteSQLClient(role="basic", seed=False)
    n = db.replace_applications([
        {"company": "Acme", "role": "GenAI Eng", "status": "Interviewing"},
        {"company": "Bolt", "role": "ML Eng", "status": "Interviewing"},
        {"company": "Cad", "role": "AI Eng", "status": "Rejected"},
    ])
    assert n == 3
    rows = await db.run_query("SELECT status, COUNT(*) AS count FROM applications GROUP BY status")
    counts = {r["status"]: r["count"] for r in rows}
    assert counts == {"Interviewing": 2, "Rejected": 1}


def test_sqlite_add_application_appends_and_dedupes():
    db = SqliteSQLClient(role="basic", seed=False)
    db.replace_applications([{"company": "Acme", "role": "GenAI Eng", "status": "Applied"}])
    assert db.count() == 1
    # a new company+role appends
    assert db.add_application({"company": "Bolt", "role": "ML Eng", "status": "Applied"}) == 2
    # an exact company+role duplicate (case-insensitive) does NOT
    assert db.add_application({"company": "bolt", "role": "ml eng", "status": "Offer"}) == 2
    # missing company/role is ignored
    assert db.add_application({"role": "Only role"}) == 2


async def test_sqlite_tracker_rejects_writes():
    db = SqliteSQLClient(role="basic")
    for bad in ("DELETE FROM applications", "UPDATE applications SET status='Offer'", "DROP TABLE applications"):
        try:
            await db.run_query(bad)
            assert False, f"write not rejected: {bad}"
        except ReadOnlyViolation:
            pass
    # the table is still intact after the rejected writes
    rows = await db.run_query("SELECT COUNT(*) AS n FROM applications")
    assert rows[0]["n"] > 0


def test_help_reply_matches_meta_and_greetings():
    assert _help_reply("what all you can do for me") is not None
    assert _help_reply("hi") is not None
    assert _help_reply("how does this work?") is not None
    # a real task must NOT be intercepted
    assert _help_reply("find senior genai roles") is None
    assert _help_reply("does my resume fit this JD?") is None
