"""A deterministic, no-network LLM/embedding stand-in so CareerPilot runs with zero keys.

This is NOT a model — it is a small rules engine that satisfies the same
``LLMClient``/``EmbeddingClient`` protocols so the demo (and a fresh deploy) works out
of the box. Set ``OPENAI_API_KEY`` to use the real OpenAI adapters instead; nothing
else changes, because everything depends on the protocols, not these classes.
"""

from __future__ import annotations

import hashlib
import math
import re

from careerpilot.core.llm_client import ToolCall, ToolResponse

_ROUTING_KEYWORDS = {
    "knowledge": ("resume", "cv", "jd", "job description", "fit", "gap", "match", "interview",
                  "behavioural", "behavioral", "negotiat", "salary", "cover letter tips"),
    "jobs": ("find", "search", "openings", "roles in", "jobs in", "hiring", "company", "postings", "listing"),
    "analytics": ("my application", "applied", "applications", "response rate", "stale", "ghosted",
                  "pending", "status of", "how many have i", "tracker"),
    "notify": ("draft", "remind", "reminder", "follow up", "follow-up", "cover letter", "email the",
               "reach out", "message the recruiter", "send"),
}


class OfflineLLMClient:
    async def get_structured_output(self, *, system_prompt, user_prompt, schema):
        name = schema.__name__
        text = user_prompt.lower()
        if name == "RoutingDecision":
            agents = [a for a, kws in _ROUTING_KEYWORDS.items() if any(k in text for k in kws)]
            if not agents:
                agents = ["knowledge"]
            return schema(agents=agents, reasoning="offline keyword routing")
        if name == "KnowledgeAnswer":
            ctx = user_prompt.split("Context:", 1)[-1].split("Question:", 1)[0].strip()
            snippet = ctx.split("\n\n")[0][:320] if ctx else ""
            return schema(
                answer=(snippet or "No relevant context was found."),
                citations=re.findall(r"\[([^\]]+)\]", ctx)[:3],
                answer_found=bool(snippet),
            )
        if name == "Draft":
            return schema(channel="email", subject="Following up on my application", body=user_prompt.strip()[:400])
        return schema.model_construct()

    async def get_tool_response(self, *, messages, tools) -> ToolResponse:
        tool_names = {t["function"]["name"] for t in tools} if tools else set()
        already_called = any(m.get("role") == "tool" for m in messages)
        user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "") or ""

        if not tools or already_called:
            return ToolResponse(content=self._summarize(messages))

        if "search_jobs" in tool_names:
            location = next((c for c in ("bangalore", "hyderabad", "remote") if c in user_text.lower()), "")
            return ToolResponse(
                content=None,
                tool_calls=[ToolCall("c1", "search_jobs", {"role": "engineer", "location": location})],
            )
        if "run_sql_query" in tool_names:
            query = "SELECT status, COUNT(*) AS count FROM applications GROUP BY status"
            return ToolResponse(content=None, tool_calls=[ToolCall("c1", "run_sql_query", {"query": query})])
        return ToolResponse(content="I don't have a tool for that.")

    @staticmethod
    def _summarize(messages) -> str:
        tool_payloads = [m["content"] for m in messages if m.get("role") == "tool"]
        if not tool_payloads:
            return "Here's what I found."
        return "Based on the data I retrieved: " + " ".join(tool_payloads)[:600]


class HashingEmbeddingClient:
    """Deterministic local embeddings (hashed bag-of-words) for offline RAG. Good enough
    for exact/near-exact retrieval in the demo; replaced by real embeddings when a key
    is set."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
