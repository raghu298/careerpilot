"""A deterministic, no-network LLM/embedding stand-in so CareerPilot runs with zero keys.

This is NOT a model — it is a small rules engine that satisfies the same
``LLMClient``/``EmbeddingClient`` protocols so the demo (and a fresh deploy) works out
of the box. Set ``OPENAI_API_KEY`` to use the real OpenAI adapters instead; nothing
else changes, because everything depends on the protocols, not these classes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from careerpilot.core.llm_client import ToolCall, ToolResponse

# A small skill vocabulary used to produce a deterministic resume-vs-JD match/gap
# analysis offline. With OPENAI_API_KEY set, the real model does this far better.
_SKILLS = [
    "python", "fastapi", "postgresql", "pgvector", "rag", "tool-calling", "agents",
    "langgraph", "kubernetes", "docker", "aws", "evaluation", "observability",
    "mlops", "llm", "sql", "retrieval",
]

_ROUTING_KEYWORDS = {
    "knowledge": ("resume", "cv", "jd", "job description", "fit", "gap", "match", "interview",
                  "behavioural", "behavioral", "negotiat", "salary", "cover letter tips"),
    "jobs": ("find", "search", "openings", "roles in", "jobs in", "hiring", "company", "postings", "listing"),
    "analytics": ("my application", "applied", "applications", "response rate", "stale", "ghosted",
                  "pending", "status of", "how many have i", "tracker"),
    "notify": ("draft", "remind", "reminder", "follow up", "follow-up", "cover letter", "email the",
               "reach out", "message the recruiter", "send"),
}


def _extract_job_query(text: str) -> tuple[str, str]:
    """Pull a search role and an optional region out of a free-text job request, so the
    live job tool is queried with the user's real intent rather than a generic term."""
    t = text.lower()
    region = next((r for r in ("worldwide", "remote", "europe", "usa", "india", "uk", "canada") if r in t), "")
    role = t
    for junk in (
        "find", "show", "get", "search", "please", "openings", "opening", "listings",
        "roles", "role", "jobs", "job", "haven't applied to yet", "haven't applied",
        "a role i", "i haven't", "applied to", "for me", "that i", " for ", " me ", " a ", " the ",
        "in bangalore", "in hyderabad", "in india", "in the us", "in europe",
        "bangalore", "hyderabad", "worldwide", "remote",
    ):
        role = role.replace(junk, " ")
    role = " ".join(role.split()).strip(" ,.")
    return (role or "software engineer"), region


class OfflineLLMClient:
    async def get_structured_output(self, *, system_prompt, user_prompt, schema):
        name = schema.__name__
        text = user_prompt.lower()
        if name == "RoutingDecision":
            agents = [a for a, kws in _ROUTING_KEYWORDS.items() if any(k in text for k in kws)]
            if not agents:
                agents = ["knowledge"]
            why = {
                "knowledge": "needs the user's resume / JD / career guides",
                "jobs": "needs live job listings or company info",
                "analytics": "needs the user's own application-tracker data",
                "notify": "needs to draft an outreach message for confirmation",
            }
            reason = "Routing to " + ", ".join(agents) + " — " + "; ".join(why[a] for a in agents) + "."
            return schema(agents=agents, reasoning=reason)
        if name == "KnowledgeAnswer":
            ctx = user_prompt.split("Context:", 1)[-1].split("Question:", 1)[0].strip()
            # Use only the user's actual question to pick an answer style — NOT the whole
            # prompt, whose retrieved context can contain words like "gaps" and wrongly
            # trigger the resume-fit branch for unrelated questions.
            question = user_prompt.split("Question:", 1)[-1].strip().lower()
            sources = re.findall(r"\[([^\]]+)\]", ctx)[:3]
            chunks = {m.group(1): m.group(2).strip() for m in re.finditer(r"\[([^\]]+)\]\s*(.*?)(?=\n\n\[|\Z)", ctx, re.S)}
            answer = self._knowledge_answer(question, chunks)
            return schema(answer=answer or "No relevant context was found.", citations=sources, answer_found=bool(answer))
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
            role, location = _extract_job_query(user_text)
            return ToolResponse(
                content=None,
                tool_calls=[ToolCall("c1", "search_jobs", {"role": role, "location": location})],
            )
        if "run_sql_query" in tool_names:
            query = "SELECT status, COUNT(*) AS count FROM applications GROUP BY status"
            return ToolResponse(content=None, tool_calls=[ToolCall("c1", "run_sql_query", {"query": query})])
        return ToolResponse(content="I don't have a tool for that.")

    @staticmethod
    def _knowledge_answer(question: str, chunks: dict) -> str:
        """Deterministic resume-vs-JD match/gap analysis, or a clean direct answer."""
        resume = next((v for k, v in chunks.items() if "my-resume" in k), "")
        jd = next((v for k, v in chunks.items() if k.startswith("jd")), "")
        wants_fit = any(w in question for w in ("fit", "gap", "match", "suit", "qualify"))

        if wants_fit and resume and jd:
            r_full, j = resume.lower(), jd.lower()
            # Skills named in the resume's own "Gaps:" clause are NOT matches.
            r_have, _, r_gap = r_full.partition("gaps:")
            jd_skills = [s for s in _SKILLS if s in j]
            matched = [s for s in jd_skills if s in r_have and s not in r_gap]
            gaps = [s for s in jd_skills if s not in matched]
            lines = ["Here's how your profile lines up with the job description:\n"]
            if matched:
                lines.append("**Strong matches** — " + ", ".join(sorted(set(matched))) + ".")
            if gaps:
                lines.append("**Gaps to address** — " + ", ".join(sorted(set(gaps))) +
                             ". Add these to your resume where genuinely true, or build a small project that shows them.")
            lines.append("\nTip: mirror the JD's exact terms in the top third of your resume when they honestly apply to you.")
            return "\n".join(lines)

        # Otherwise: a clean direct answer from the most relevant chunk (no [tag] noise).
        if chunks:
            return next(iter(chunks.values()))
        return ""

    @staticmethod
    def _summarize(messages) -> str:
        payloads = []
        for m in messages:
            if m.get("role") == "tool":
                try:
                    payloads.append(json.loads(m["content"]))
                except (ValueError, TypeError):
                    pass
        if not payloads:
            return "Here's what I found."

        out: list[str] = []
        for p in payloads:
            data = p.get("rows") if isinstance(p, dict) and "rows" in p else p
            if isinstance(p, dict) and "jobs" in p:
                jobs = p["jobs"]
                if not jobs:
                    out.append("I couldn't find live roles matching that right now — try a broader title.")
                    continue
                src = " (live)" if str(p.get("source", "")).startswith("remotive") else ""
                out.append(f"I found {len(jobs)} matching role(s){src}:")
                for j in jobs:
                    line = f"• {j.get('title')} at {j.get('company')} ({j.get('location')})"
                    if j.get("stack"):
                        line += f" — {j['stack']}"
                    salary = j.get("salary") or (f"₹{j['salary_lpa']} LPA" if j.get("salary_lpa") else "")
                    if salary:
                        line += f" · {salary}"
                    if j.get("url"):
                        line += f"\n  Apply: {j['url']}"
                    out.append(line)
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                if {"status", "count"} <= set(data[0]):
                    out.append("Your applications by status: " +
                               ", ".join(f"{r['status']} ({r['count']})" for r in data) + ".")
                else:
                    out.append("Here's what the tracker shows:")
                    for row in data[:10]:
                        out.append("• " + ", ".join(f"{k}: {v}" for k, v in row.items()))
            elif isinstance(p, dict) and p.get("name"):
                out.append(f"{p.get('name')} — {p.get('industry', '')}, ~{p.get('size', '?')} people. {p.get('summary', '')}")
            else:
                out.append(str(p))
        return "\n".join(out)


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
