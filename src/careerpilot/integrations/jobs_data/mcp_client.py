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
        if name == "get_company":
            key = str(arguments.get("name", "")).lower().strip()
            profile = _COMPANIES.get(key)
            return profile or {"name": arguments.get("name"), "error": "unknown company"}
        return {"error": f"unknown tool {name}"}


class MCPJobsSource:
    """Real adapter: proxies tool calls to an external jobs MCP server.

    The session is opened once per process (by ``bootstrap.build_shared_resources``)
    and closed on shutdown. ``call_tool`` returns the server's structured content.
    """

    def __init__(self, session) -> None:  # an initialized mcp.ClientSession
        self._session = session

    def tool_specs(self) -> list[dict]:
        return _TOOL_SPECS

    async def call_tool(self, name: str, arguments: dict) -> Any:
        result = await self._session.call_tool(name, arguments)
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        parts = getattr(result, "content", []) or []
        texts = [getattr(p, "text", "") for p in parts if getattr(p, "type", None) == "text"]
        return {"text": "\n".join(t for t in texts if t)}


async def open_mcp_jobs_session(server_url: str, token: str | None = None):
    """Open an MCP client session to a jobs MCP server (streamable HTTP).

    Returns ``(session, aclose)``. Imported lazily so the ``mcp`` SDK is only required
    when a real server is actually configured.
    """
    from contextlib import AsyncExitStack

    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    stack = AsyncExitStack()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    read, write, _ = await stack.enter_async_context(streamablehttp_client(server_url, headers=headers))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    async def aclose() -> None:
        await stack.aclose()

    return session, aclose
