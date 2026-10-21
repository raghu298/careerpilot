"""Jobs-data seam, reached over the Model Context Protocol (MCP).

Mirrors NexusAI's Atlassian-over-MCP integration: the Jobs Agent never calls a
job-board API directly. It asks a ``JobsSource`` for named tools and invokes them, and
the real adapter (``MCPJobsSource``) speaks MCP to an external jobs MCP server using the
official ``mcp`` SDK. The fake (``FakeJobsMCP``) answers the same tool calls from canned
data so the agent is fully testable offline.

Tools exposed by the server (and faithfully mimicked by the fake):
  - search_jobs(role, location)  -> matching live openings
  - get_company(name)            -> short company profile
"""

from __future__ import annotations

from typing import Any, Protocol


class JobsSource(Protocol):
    def tool_specs(self) -> list[dict]:
        """OpenAI-style function specs the agent offers to the LLM."""
        ...

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Invoke one MCP tool and return its (JSON-serializable) result."""
        ...


_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search live job postings by role and (optional) location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "e.g. 'backend engineer', 'data scientist'"},
                    "location": {"type": "string", "description": "e.g. 'Bangalore', 'remote'"},
                },
                "required": ["role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "salary_insights",
            "description": "Typical (median) salary range for a role, aggregated from live salary-disclosed postings. Use for 'what do X roles pay', 'salary range for X', 'market rate'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "e.g. 'senior genai engineer'"},
                    "location": {"type": "string"},
                },
                "required": ["role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trending_skills",
            "description": "Most in-demand skills for a role, by frequency across live postings. Use for 'what skills are in demand / should I learn for X roles'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "e.g. 'genai engineer'"},
                    "location": {"type": "string"},
                },
                "required": ["role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company",
            "description": "Short profile for a company (industry, size, what they build).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
]

_JOBS = [
    {"title": "Senior GenAI Engineer", "company": "Nimbus Labs", "location": "Bangalore", "stack": "Python, LangGraph, RAG", "salary_lpa": 42},
    {"title": "Backend Engineer (AI)", "company": "Flowbit", "location": "Bangalore", "stack": "Python, FastAPI, Postgres", "salary_lpa": 34},
    {"title": "Staff ML Engineer", "company": "Coreline", "location": "Remote", "stack": "Python, LLMs, MLOps", "salary_lpa": 55},
    {"title": "AI Platform Engineer", "company": "Nimbus Labs", "location": "Hyderabad", "stack": "Python, Kubernetes, vLLM", "salary_lpa": 46},
]

_COMPANIES = {
    "nimbus labs": {"name": "Nimbus Labs", "industry": "AI infrastructure", "size": "120", "summary": "Builds agentic developer tooling on top of LLMs."},
    "flowbit": {"name": "Flowbit", "industry": "Fintech", "size": "340", "summary": "Payments platform adding AI assistants to its product."},
    "coreline": {"name": "Coreline", "industry": "Enterprise SaaS", "size": "900", "summary": "Document intelligence and retrieval for large enterprises."},
}


class FakeJobsMCP:
    """Deterministic offline stand-in for the jobs MCP server."""

    def tool_specs(self) -> list[dict]:
        return _TOOL_SPECS

    async def call_tool(self, name: str, arguments: dict) -> Any:
        if name == "search_jobs":
            role = str(arguments.get("role", "")).lower()
            loc = str(arguments.get("location", "")).lower()
            matches = [
                j
                for j in _JOBS
                if (not role or any(w in j["title"].lower() or w in j["stack"].lower() for w in role.split()))
                and (not loc or loc in j["location"].lower())
            ]
            return {"count": len(matches), "jobs": matches or _JOBS}
        if name == "salary_insights":
            role = str(arguments.get("role", "")).lower()
            loc = str(arguments.get("location", "")).lower()
            matches = [
                j for j in _JOBS
                if (not role or any(w in j["title"].lower() or w in j["stack"].lower() for w in role.split()))
                and (not loc or loc in j["location"].lower())
            ] or _JOBS
            lpas = sorted(j["salary_lpa"] for j in matches)
            mid = len(lpas) // 2
            median = lpas[mid] if len(lpas) % 2 else (lpas[mid - 1] + lpas[mid]) / 2
            return {
                "role": arguments.get("role", ""),
                "sample_size": len(lpas),
                "typical_range": f"₹{median:g} LPA",
                "low": f"₹{min(lpas)} LPA",
                "high": f"₹{max(lpas)} LPA",
                "basis": "median of sample postings",
                "source": "fake (offline)",
            }
        if name == "trending_skills":
            role = str(arguments.get("role", "")).lower()
            loc = str(arguments.get("location", "")).lower()
            matches = [
                j for j in _JOBS
                if (not role or any(w in j["title"].lower() or w in j["stack"].lower() for w in role.split()))
                and (not loc or loc in j["location"].lower())
            ] or _JOBS
            counts: dict[str, int] = {}
            for j in matches:
                for skill in {s.strip().lower() for s in j["stack"].split(",") if s.strip()}:
                    counts[skill] = counts.get(skill, 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            return {
                "role": arguments.get("role", ""),
                "postings_analysed": len(matches),
                "skills": [{"skill": s, "count": n} for s, n in top],
                "source": "fake (offline)",
            }
        if name == "get_company":
            key = str(arguments.get("name", "")).lower().strip()
            profile = _COMPANIES.get(key)
            return profile or {"name": arguments.get("name"), "error": "unknown company"}
        return {"error": f"unknown tool {name}"}


def _parse_tool_result(result: Any) -> Any:
    import json

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps a non-model return under a "result" key; unwrap it.
        return structured.get("result", structured)
    parts = getattr(result, "content", []) or []
    texts = [getattr(p, "text", "") for p in parts if getattr(p, "type", None) == "text"]
    blob = "\n".join(t for t in texts if t)
    try:
        return json.loads(blob)  # tool returned JSON text — parse so the agent sees fields
    except (ValueError, TypeError):
        return {"text": blob}


class MCPJobsSource:
    """Real adapter: a genuine MCP client that opens a fresh session per call.

    A per-call session (rather than a long-lived one) keeps every MCP interaction
    inside the task that issues it, which avoids anyio cancel-scope errors when the
    call originates from within LangGraph's node execution. Transport is either a
    stdio subprocess running the bundled server (``script_path``) or a remote
    streamable-HTTP server (``url``).
    """

    def __init__(self, *, script_path: str | None = None, url: str | None = None, token: str | None = None) -> None:
        self._script_path = script_path
        self._url = url
        self._token = token

    def tool_specs(self) -> list[dict]:
        return _TOOL_SPECS

    async def call_tool(self, name: str, arguments: dict) -> Any:
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        stack = AsyncExitStack()
        try:
            if self._url:
                from mcp.client.streamable_http import streamablehttp_client

                headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
                read, write, _ = await stack.enter_async_context(streamablehttp_client(self._url, headers=headers))
            else:
                import sys

                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(command=sys.executable, args=[self._script_path])
                read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            result = await session.call_tool(name, arguments)
            return _parse_tool_result(result)
        finally:
            await stack.aclose()
