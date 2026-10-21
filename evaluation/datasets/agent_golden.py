"""Golden datasets for the Jobs and Analytics agents' correctness evaluation.

These are deterministic, assertion-style expectations (no LLM judge required) that
encode real behaviours the agents must preserve — again, several are regressions for
bugs found during iterative testing (irrelevant job ranking, router cross-talk).
"""

from __future__ import annotations

# Each case runs a job search and asserts properties of the ranked results. We avoid
# asserting exact titles (live data changes daily) and instead assert topical relevance:
# at least one of `expect_any_term` must appear in the top results' title+stack.
JOBS_GOLDEN_CASES: list[dict] = [
    {
        "query": "find senior GenAI engineer jobs",
        "expect_terms": ["ai", "ml", "machine learning", "llm", "data", "genai"],
        "min_results": 1,
        "note": "GenAI search must return AI/ML-relevant roles, not generic 'senior' roles (regression).",
    },
    {
        "query": "find machine learning engineer jobs",
        "expect_terms": ["ai", "ml", "machine learning", "data", "learning"],
        "min_results": 1,
        "note": "ML search relevance.",
    },
    {
        "query": "find python backend developer roles",
        "expect_terms": ["python", "backend", "api", "django", "fastapi", "engineer"],
        "min_results": 1,
        "note": "Backend search relevance.",
    },
]

# Analytics cases assert the agent produces the right read-only SQL summary over the
# fake application tracker (status counts), and that writes are refused.
ANALYTICS_GOLDEN_CASES: list[dict] = [
    {
        "query": "how many applications have I sent, by status?",
        "expect_substrings": ["applied", "status"],
        "note": "NL->SQL status rollup over the application tracker.",
    },
    {
        "query": "which of my applications are still pending?",
        "expect_substrings": ["interviewing"],  # pending => Applied/Screening/Interviewing
        "note": "Status filter — 'pending' maps to in-progress statuses.",
    },
    {
        "query": "which applications have I been ghosted on?",
        "expect_substrings": ["ghosted"],
        "note": "Status filter — ghosted roles.",
    },
]
