"""A real MCP server exposing live job search for CareerPilot's Jobs agent.

Backed by the free Remotive API (https://remotive.com/api/remote-jobs) — real remote
job postings, no API key required. The CareerPilot backend launches this as a stdio
MCP subprocess and calls its tools over the Model Context Protocol, so the Jobs agent
makes genuine MCP calls against live data.

Tools:
  - search_jobs(role, location) -> live matching postings
  - get_company(name)           -> a lightweight profile from that company's live postings
"""

from __future__ import annotations

import re

import httpx
from mcp.server.mcpserver import MCPServer

_STOP = {"the", "for", "and", "a", "in", "at", "of", "to", "with", "role", "roles", "job", "jobs"}

mcp = MCPServer("careerpilot-jobs")

_API = "https://remotive.com/api/remote-jobs"


def _fetch(search: str, limit: int = 20) -> list[dict]:
    with httpx.Client(timeout=15) as client:
        resp = client.get(_API, params={"search": search, "limit": limit})
        resp.raise_for_status()
        return resp.json().get("jobs", [])


def _shape(job: dict) -> dict:
    return {
        "title": job.get("title"),
        "company": job.get("company_name"),
        "location": job.get("candidate_required_location") or "Remote",
        "stack": ", ".join((job.get("tags") or [])[:4]),
        "salary": job.get("salary") or "",
        "url": job.get("url"),
    }


@mcp.tool()
def search_jobs(role: str, location: str = "") -> dict:
    """Search live remote job postings by role and optional location."""
    try:
        raw = _fetch(role, limit=25)
    except Exception as exc:  # degrade gracefully; the agent still answers
        return {"count": 0, "jobs": [], "error": f"job source unavailable: {exc}"}

    loc = location.lower().strip()
    terms = [w for w in re.findall(r"[a-z0-9+#]+", role.lower()) if w not in _STOP and len(w) > 1]

    scored: list[tuple[int, dict]] = []
    for job in raw:
        where = (job.get("candidate_required_location") or "").lower()
        if loc and loc not in where and "worldwide" not in where and "anywhere" not in where:
            continue
        hay = (job.get("title", "") + " " + " ".join(job.get("tags") or [])).lower()
        score = sum(1 for t in terms if t in hay)
        if score > 0:
            scored.append((score, job))

    scored.sort(key=lambda x: -x[0])
    jobs = [_shape(j) for _, j in scored[:8]]
    if not jobs:  # nothing matched well — show the top live results for the raw search
        jobs = [_shape(j) for j in raw[:6]]
    return {"count": len(jobs), "jobs": jobs, "source": "remotive (live)"}


@mcp.tool()
def get_company(name: str) -> dict:
    """A lightweight company profile derived from that company's live postings."""
    try:
        raw = _fetch(name, limit=25)
    except Exception as exc:
        return {"name": name, "error": f"job source unavailable: {exc}"}

    postings = [_shape(j) for j in raw if name.lower() in (j.get("company_name") or "").lower()]
    if not postings:
        return {"name": name, "summary": "No live postings found for this company right now."}
    return {
        "name": postings[0]["company"],
        "open_roles": len(postings),
        "sample_roles": [p["title"] for p in postings[:5]],
        "source": "remotive (live)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
