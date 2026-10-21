"""Tests for the jobs MCP server's pure ranking/salary helpers.

These guard the relevance + salary-hygiene rules that make live job results useful:
real engineering roles outrank gig/micro-task listings, and misleading salary strings
(per-task/degenerate ranges) are never shown. The server module is a top-level script,
so we add its directory to the path before importing its pure functions (no network).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parents[2] / "mcp_server"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import jobs_mcp_server as m  # noqa: E402


def test_salary_range_suppresses_degenerate_and_low():
    assert m._salary_range(120_000, 180_000) == "$120k–$180k"
    assert m._salary_range(10_000, 10_000) == ""      # degenerate (min == max)
    assert m._salary_range(500, 900) == ""            # implausibly low
    assert m._salary_range(None, 180_000) == ""


def test_clean_salary_drops_gig_pay():
    assert m._clean_salary("$1000 per task") == ""
    assert m._clean_salary("$40/hr") == ""
    assert m._clean_salary("$120k–$180k") == "$120k–$180k"


def test_rank_puts_real_engineer_above_gig():
    pool = [
        {"title": "Software Engineer AI Coding Tasks", "company": "X", "location": "Remote",
         "tags": ["ai"], "description": "", "url": "u1", "salary": "$1000 per task"},
        {"title": "Machine Learning Engineer", "company": "Y", "location": "Remote",
         "tags": ["ml", "ai"], "description": "machine learning", "url": "u2", "salary": ""},
    ]
    ranked = m._rank(["ai", "machine learning"], pool, "", ["engineer"])
    titles = [j["title"] for j in ranked]
    assert titles.index("Machine Learning Engineer") < titles.index("Software Engineer AI Coding Tasks")


def test_rank_whole_word_match_not_substring():
    """'ai' must not match 'Rails'/'brain' — the whole-word matcher guards relevance."""
    pool = [
        {"title": "Rails Developer", "company": "X", "location": "Remote",
         "tags": ["rails"], "description": "ruby on rails", "url": "u1", "salary": ""},
    ]
    ranked = m._rank(["ai"], pool, "", [])
    assert ranked == []  # no genuine 'ai' token present
