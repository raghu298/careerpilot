"""Correctness evaluation for the Jobs and Analytics agents.

Deterministic, assertion-style checks (no LLM judge) against
``evaluation/datasets/agent_golden.py``. Jobs cases assert topical relevance of the
ranked live results (not exact titles, since live data changes); Analytics cases assert
the NL->SQL rollup content and the read-only guardrail.

Usage:  python -m evaluation.run_agent_eval
"""

from __future__ import annotations

import asyncio
import sys

from careerpilot.access_policy import resolve_capabilities
from careerpilot.agents.analytics.agent import AnalyticsAgent
from careerpilot.agents.jobs.agent import JobsAgent
from careerpilot.bootstrap import build_shared_resources
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient
from careerpilot.integrations.sql_db.postgres_client import ReadOnlyViolation
from evaluation.datasets.agent_golden import ANALYTICS_GOLDEN_CASES, JOBS_GOLDEN_CASES


async def main() -> int:
    resources = await build_shared_resources()
    print(f"=== Agent eval — jobs source: {resources.jobs_mode} ===\n")
    caps = resolve_capabilities("arjun")  # full access
    failures: list[str] = []

    # ---- Jobs ----
    print("--- Jobs agent (relevance) ---")
    jobs_agent = JobsAgent(resources.llm, resources.jobs_source, caps)
    for case in JOBS_GOLDEN_CASES:
        result = await jobs_agent.handle(case["query"])
        text = result.content.lower()
        n_bullets = text.count("•")
        hit = any(t in text for t in case["expect_terms"])
        status = "✓" if hit and n_bullets >= case["min_results"] else "✗"
        print(f"  {status} {case['query']!r}  (bullets={n_bullets}, relevant_term={hit})")
        if status == "✗":
            failures.append(f"[jobs] {case['query']!r}: relevance/term check failed")
        for line in result.content.splitlines()[1:4]:
            print(f"        {line.strip()}")
    print()

    # ---- Analytics ----
    print("--- Analytics agent (NL->SQL) ---")
    analytics = AnalyticsAgent(resources.llm, SqliteSQLClient(role=caps.sql_role), caps)
    for case in ANALYTICS_GOLDEN_CASES:
        result = await analytics.handle(case["query"])
        text = result.content.lower()
        ok = all(s in text for s in case["expect_substrings"])
        status = "✓" if ok else "✗"
        print(f"  {status} {case['query']!r}")
        print(f"        {result.content.splitlines()[0][:120]}")
        if not ok:
            failures.append(f"[analytics] {case['query']!r}: missing {case['expect_substrings']}")
    print()

    # ---- read-only guardrail (same path the real adapter uses) ----
    print("--- Analytics read-only guardrail ---")
    for bad_sql in ("DELETE FROM applications", "UPDATE applications SET status='Offer'",
                    "DROP TABLE applications"):
        try:
            await SqliteSQLClient(role="basic").run_query(bad_sql)
            failures.append(f"[guardrail] write NOT rejected: {bad_sql!r}")
            print(f"  ✗ allowed: {bad_sql!r}")
        except ReadOnlyViolation as exc:  # guardrail rejects non-SELECT
            print(f"  ✓ rejected {bad_sql.split()[0]}: {exc}")
    print()

    if failures:
        print(f"✗ {len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✓ All agent eval checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
