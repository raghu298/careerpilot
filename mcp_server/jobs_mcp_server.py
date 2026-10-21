"""A real MCP server exposing live job search for CareerPilot's Jobs agent.

Backed by free, no-key job feeds:
  - RemoteOK   (https://remoteok.com/api)            — tech-tagged remote roles (primary)
  - Remotive   (https://remotive.com/api/remote-jobs) — broader remote roles  (backfill)

Both are real live data and need no API key. RemoteOK is tech-focused and curated, so
it gives far better signal for specialised searches (GenAI/ML/data) than a general
feed. The CareerPilot backend launches this as a stdio MCP subprocess and calls its
tools over the Model Context Protocol, so the Jobs agent makes genuine MCP calls.

Tools:
  - search_jobs(role, location) -> live matching postings, ranked
  - get_company(name)           -> a lightweight profile from that company's live postings
"""

from __future__ import annotations

import re

import httpx
from mcp.server.mcpserver import MCPServer

# Seniority / filler words carry no topical signal — dropping them stops a search
# like "senior genai" from collapsing onto every "senior <anything>" posting.
_STOP = {
    "the", "for", "and", "a", "in", "at", "of", "to", "with", "role", "roles", "job", "jobs",
    "senior", "junior", "lead", "staff", "principal", "sr", "jr", "mid", "level", "entry",
    "engineer", "developer", "position", "opening", "openings", "remote",
}

# Expand the user's topic into the terms a job board actually indexes/tags.
_SYNONYMS = {
    "genai": ["ai", "llm", "generative", "machine learning", "nlp"],
    "gen": ["ai"],
    "ml": ["machine learning", "ai"],
    "ai": ["ai", "machine learning", "llm"],
    "llm": ["llm", "ai", "nlp"],
    "nlp": ["nlp", "machine learning", "ai"],
    "ds": ["data science", "data"],
    "fullstack": ["full stack", "fullstack"],
    "frontend": ["frontend", "react"],
    "backend": ["backend", "api"],
    "devops": ["devops", "kubernetes", "infrastructure"],
}

mcp = MCPServer("careerpilot-jobs")

_REMOTEOK = "https://remoteok.com/api"
_REMOTIVE = "https://remotive.com/api/remote-jobs"
_UA = {"User-Agent": "careerpilot-jobs (+https://github.com/raghu298/careerpilot)"}


# Multi-word topics matched as a phrase, so "learning" alone doesn't match soft fluff
# ("a great learning environment") and inflate unrelated roles.
_PHRASES = {
    "machine learning": ["machine learning", "ml", "ai"],
    "data science": ["data science", "data scientist"],
    "full stack": ["full stack", "fullstack"],
    "product manager": ["product manager", "product management"],
}


def _topic_terms(role: str) -> list[str]:
    """Meaningful, synonym-expanded terms used both to query and to rank."""
    low = role.lower()
    terms: list[str] = []
    consumed = low
    for phrase, expand in _PHRASES.items():
        if phrase in low:
            terms.extend(expand)
            consumed = consumed.replace(phrase, " ")
    raw = [w for w in re.findall(r"[a-z0-9+#]+", consumed) if w not in _STOP and len(w) > 1]
    for w in raw:
        terms.extend(_SYNONYMS.get(w, [w]))
    seen: set[str] = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


def _norm(*, title, company, location, tags, description, url, salary="", source) -> dict:
    """Common internal record so both feeds rank identically."""
    return {
        "title": title or "",
        "company": company or "",
        "location": location or "Remote",
        "tags": [str(t) for t in (tags or [])],
        "description": description or "",
        "url": url or "",
        "salary": salary or "",
        "source": source,
    }


def _fetch_remoteok() -> list[dict]:
    with httpx.Client(timeout=15, headers=_UA) as client:
        data = client.get(_REMOTEOK).json()
    return [
        _norm(
            title=j.get("position"), company=j.get("company"), location=j.get("location"),
            tags=j.get("tags"), description=j.get("description"), url=j.get("url"),
            salary=_salary_range(j.get("salary_min"), j.get("salary_max")), source="remoteok",
        )
        for j in data
        if isinstance(j, dict) and j.get("position")
    ]


def _fetch_remotive(search: str, limit: int = 50) -> list[dict]:
    with httpx.Client(timeout=15, headers=_UA) as client:
        resp = client.get(_REMOTIVE, params={"search": search, "limit": limit})
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
    return [
        _norm(
            title=j.get("title"), company=j.get("company_name"),
            location=j.get("candidate_required_location"), tags=j.get("tags"),
            description=j.get("description"), url=j.get("url"),
            salary=j.get("salary"), source="remotive",
        )
        for j in jobs
    ]


def _salary_range(lo, hi) -> str:
    if lo and hi:
        return f"${int(lo/1000)}k–${int(hi/1000)}k"
    return ""


def _shape(job: dict) -> dict:
    return {
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "stack": ", ".join(job["tags"][:4]),
        "salary": job["salary"],
        "url": job["url"],
        "via": job["source"],
    }


def _rank(terms: list[str], pool: list[dict], location: str) -> list[dict]:
    # Whole-word matchers — critical so "ai" doesn't match "Rails" or "go" "Django".
    matchers = [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms]
    loc = location.lower().strip()
    seen_urls: set[str] = set()
    scored: list[tuple[int, dict]] = []
    for job in pool:
        url = job["url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        where = job["location"].lower()
        if loc and loc not in where and "worldwide" not in where and "anywhere" not in where:
            continue
        title = job["title"].lower()
        tags = " ".join(job["tags"]).lower()
        desc = re.sub(r"<[^>]+>", " ", job["description"]).lower()
        # Title hit ≫ tag hit ≫ description hit.
        score = sum(
            5 * bool(m.search(title)) + 2 * bool(m.search(tags)) + 1 * bool(m.search(desc))
            for m in matchers
        )
        if score > 0:
            scored.append((score, job))
    scored.sort(key=lambda x: -x[0])
    return [j for _, j in scored]


@mcp.tool()
def search_jobs(role: str, location: str = "") -> dict:
    """Search live remote job postings by role and optional location."""
    terms = _topic_terms(role)
    pool: list[dict] = []
    errors: list[str] = []

    # Primary: RemoteOK (tech-tagged, curated).
    try:
        pool += _fetch_remoteok()
    except Exception as exc:
        errors.append(f"remoteok: {exc}")

    # Backfill: Remotive full-text search on the strongest term, for volume/coverage.
    try:
        pool += _fetch_remotive(terms[0] if terms else role.strip())
    except Exception as exc:
        errors.append(f"remotive: {exc}")

    if not pool:
        return {"count": 0, "jobs": [], "error": "; ".join(errors) or "no live job source reachable"}

    ranked = _rank(terms, pool, location)
    jobs = [_shape(j) for j in ranked[:8]]
    if not jobs:
        # Nothing matched the topic — surface the freshest tech roles rather than nothing.
        jobs = [_shape(j) for j in pool[:6]]
    return {
        "count": len(jobs), "jobs": jobs, "terms": terms,
        "source": "remoteok + remotive (live)",
    }


@mcp.tool()
def get_company(name: str) -> dict:
    """A lightweight company profile derived from that company's live postings."""
    pool: list[dict] = []
    try:
        pool += _fetch_remoteok()
    except Exception:
        pass
    try:
        pool += _fetch_remotive(name, limit=30)
    except Exception as exc:
        if not pool:
            return {"name": name, "error": f"job source unavailable: {exc}"}

    postings = [_shape(j) for j in pool if name.lower() in j["company"].lower()]
    if not postings:
        return {"name": name, "summary": "No live postings found for this company right now."}
    return {
        "name": postings[0]["company"],
        "open_roles": len(postings),
        "sample_roles": [p["title"] for p in postings[:5]],
        "source": "remoteok + remotive (live)",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
