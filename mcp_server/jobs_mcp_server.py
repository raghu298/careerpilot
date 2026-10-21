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


def _norm(*, title, company, location, tags, description, url, salary="",
          salary_min=None, salary_max=None, source) -> dict:
    """Common internal record so both feeds rank identically."""
    return {
        "title": title or "",
        "company": company or "",
        "location": location or "Remote",
        "tags": [str(t) for t in (tags or [])],
        "description": description or "",
        "url": url or "",
        "salary": salary or "",
        "salary_min": salary_min,  # numeric (RemoteOK) — used for salary aggregation
        "salary_max": salary_max,
        "source": source,
    }


def _fetch_remoteok() -> list[dict]:
    with httpx.Client(timeout=15, headers=_UA) as client:
        data = client.get(_REMOTEOK).json()
    return [
        _norm(
            title=j.get("position"), company=j.get("company"), location=j.get("location"),
            tags=j.get("tags"), description=j.get("description"), url=j.get("url"),
            salary=_salary_range(j.get("salary_min"), j.get("salary_max")),
            salary_min=j.get("salary_min"), salary_max=j.get("salary_max"), source="remoteok",
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
    # Suppress degenerate ($10k–$10k) and implausibly-low (per-task placeholder) ranges
    # so the UI never shows a misleading "salary".
    if lo and hi and lo >= 10_000 and hi > lo:
        return f"${int(lo/1000)}k–${int(hi/1000)}k"
    return ""


# Free-text salary markers that mean "not an annual salary" — gig/hourly/per-task pay
# that would be misleading shown as a role's compensation.
_GIG_PAY = ("per task", "per hour", "/hr", "hourly", "per word", "per project", "per month")


def _clean_salary(salary: str) -> str:
    return "" if any(m in salary.lower() for m in _GIG_PAY) else salary


def _shape(job: dict) -> dict:
    return {
        "title": job["title"],
        "company": job["company"],
        "location": job["location"],
        "stack": ", ".join(job["tags"][:4]),
        "salary": _clean_salary(job["salary"]),
        "url": job["url"],
        "via": job["source"],
    }


# The kind of role the user asked for (stripped from topic terms, but it matters a lot
# for relevance: an "ML engineer" search should rank Engineer titles above "AI Trainer"
# or "Product Manager, AI"). A title in the requested kind is boosted; an off-kind title
# (manager/trainer/sales/support when an engineer/dev/scientist was asked) is penalised.
_ROLE_KINDS = ("engineer", "developer", "scientist", "analyst", "architect", "researcher",
               "designer", "manager")
_OFFKIND = ("trainer", "manager", "sales", "support", "recruiter", "assistant", "coordinator",
            "annotator", "reviewer", "writer", "marketing")
# Gig / micro-task / data-labeling listings — real postings, but almost never what
# someone searching for an "ML engineer" wants, so they're pushed down regardless of
# whether the title happens to contain a role-kind word like "engineer".
_GIG = ("coding task", "ai coding", "data annotation", "annotation", "data labeling",
        "labeling", "labelling", "micro task", "microtask", "per task")

# RemoteOK tags every posting with broad category/meta labels (seniority, department,
# perks, industry) that aren't *skills to learn*. Excluding them keeps trending_skills
# focused on real, learnable skills rather than "exec" or "digital nomad".
_NON_SKILL_TAGS = {
    "exec", "senior", "junior", "lead", "non tech", "digital nomad", "remote", "full time",
    "part time", "contract", "internship", "ops", "operations", "hr", "legal", "finance",
    "sales", "marketing", "customer support", "support", "medical", "healthcare", "education",
    "teacher", "teaching", "travel", "adult", "crypto", "web3", "saas", "growth", "seo",
    "content", "content writing", "copywriting", "video", "english", "stats", "admin",
    "quality assurance", "recruiter", "recruiting", "management", "manager", "design",
    "designer", "product", "strategy", "consulting", "nonprofit", "ecommerce", "fintech",
    "excel", "microsoft", "word", "powerpoint", "technical", "dev", "software",
}


def _rank(terms: list[str], pool: list[dict], location: str, role_kinds: list[str]) -> list[dict]:
    # Whole-word matchers — critical so "ai" doesn't match "Rails" or "go" "Django".
    matchers = [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms]
    kind_res = [re.compile(r"\b" + re.escape(k) + r"\b") for k in role_kinds]
    technical_kind = any(k in ("engineer", "developer", "scientist", "architect") for k in role_kinds)
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
            # Boost a title that matches the requested role kind.
            if any(k.search(title) for k in kind_res):
                score += 6
            # Penalise an off-kind title when a technical role was requested (e.g. a
            # "Trainer"/"Manager" surfacing for an "engineer" search).
            elif technical_kind and any(w in title for w in _OFFKIND):
                score -= 3
            # Gig / micro-task listings get pushed down even if the title says "engineer".
            haystack = title + " " + tags + " " + job.get("salary", "").lower()
            if any(w in haystack for w in _GIG):
                score -= 7
            scored.append((score, job))
    scored.sort(key=lambda x: -x[0])
    return [j for _, j in scored]


@mcp.tool()
def search_jobs(role: str, location: str = "") -> dict:
    """Search live remote job postings by role and optional location."""
    terms = _topic_terms(role)
    role_kinds = [k for k in _ROLE_KINDS if re.search(r"\b" + k + r"\b", role.lower())]
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

    ranked = _rank(terms, pool, location, role_kinds)
    jobs = [_shape(j) for j in ranked[:8]]
    if not jobs:
        # Nothing matched the topic — surface the freshest tech roles rather than nothing.
        jobs = [_shape(j) for j in pool[:6]]
    return {
        "count": len(jobs), "jobs": jobs, "terms": terms,
        "source": "remoteok + remotive (live)",
    }


def _median(nums: list[float]) -> float:
    s = sorted(nums)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


@mcp.tool()
def salary_insights(role: str, location: str = "") -> dict:
    """Aggregate real salary ranges for a role from live salary-tagged postings.

    Only RemoteOK postings carry structured salary numbers, so the sample is drawn from
    the postings that actually match the role AND disclose pay — the figures are real,
    never invented. Returns the typical (median) band plus the low/high across the
    sample so the user sees the spread, not a single made-up number.
    """
    terms = _topic_terms(role)
    role_kinds = [k for k in _ROLE_KINDS if re.search(r"\b" + k + r"\b", role.lower())]
    pool: list[dict] = []
    try:
        pool += _fetch_remoteok()
    except Exception as exc:
        return {"role": role, "sample_size": 0, "error": f"salary source unavailable: {exc}"}

    ranked = _rank(terms, pool, location, role_kinds)[:60]
    # Keep only plausible annual-salary pairs — RemoteOK mixes in per-task/hourly and
    # placeholder figures that would otherwise drag the range to "$0k" or a huge outlier.
    pairs = [
        (j["salary_min"], j["salary_max"]) for j in ranked
        if j.get("salary_min") and j.get("salary_max")
        # Floor at $20k/yr: below that is almost always a per-task/part-time placeholder
        # that would drag the reported band misleadingly low.
        and 20_000 <= j["salary_min"] <= 1_000_000 and j["salary_max"] <= 2_000_000
        and j["salary_max"] >= j["salary_min"]
    ]
    lows = [p[0] for p in pairs]
    highs = [p[1] for p in pairs]
    if not lows:
        return {
            "role": role, "sample_size": 0,
            "note": "No salary-disclosed postings matched that role right now — try a broader title.",
        }
    k = lambda v: f"${int(v / 1000)}k"  # noqa: E731
    return {
        "role": role,
        "sample_size": len(lows),
        "typical_range": f"{k(_median(lows))}–{k(_median(highs))}",
        "low": k(min(lows)),
        "high": k(max(highs)),
        "basis": "median of live salary-disclosed postings",
        "source": "remoteok (live)",
    }


@mcp.tool()
def trending_skills(role: str, location: str = "") -> dict:
    """The skills most in demand for a role right now, by frequency across live postings.

    Counts the tags/keywords on the postings that actually match the role, so the answer
    reflects what employers are asking for today — real aggregation, not a canned list.
    """
    terms = _topic_terms(role)
    role_kinds = [k for k in _ROLE_KINDS if re.search(r"\b" + k + r"\b", role.lower())]
    pool: list[dict] = []
    try:
        pool += _fetch_remoteok()
    except Exception as exc:
        errors = str(exc)
    else:
        errors = ""
    try:
        pool += _fetch_remotive(terms[0] if terms else role.strip())
    except Exception:
        pass
    if not pool:
        return {"role": role, "count": 0, "error": errors or "no live job source reachable"}

    ranked = _rank(terms, pool, location, role_kinds)[:60]
    if not ranked:
        return {"role": role, "count": 0, "note": "No matching live postings to analyse right now."}

    counts: dict[str, int] = {}
    # Query terms themselves aren't "skills to learn" — drop them so results are additive;
    # also drop RemoteOK's category/meta tags that aren't learnable skills.
    noise = set(terms) | _NON_SKILL_TAGS | {"engineer", "developer"}
    for job in ranked:
        for tag in {t.strip().lower() for t in job["tags"] if t.strip()}:
            if len(tag) < 2 or tag in noise:
                continue
            counts[tag] = counts.get(tag, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    if not top:
        return {"role": role, "count": len(ranked), "note": "Matching postings had no skill tags to rank."}
    return {
        "role": role,
        "postings_analysed": len(ranked),
        "skills": [{"skill": s, "count": n} for s, n in top],
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
