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


# Generic words that carry no topic — excluded when checking whether a retrieved chunk
# is actually on-topic for the question (lexical-grounding gate below).
_QUESTION_STOP = {
    "how", "the", "and", "for", "with", "what", "are", "can", "should", "would", "could",
    "does", "did", "will", "about", "into", "this", "that", "these", "those", "your", "you",
    "our", "their", "them", "they", "any", "all", "some", "when", "where", "which", "who",
    "whom", "why", "best", "good", "get", "got", "give", "want", "need", "like", "have",
    "has", "had", "been", "being", "from", "out", "off", "day", "days", "now", "one", "two",
}


def _build_sql(question: str) -> str:
    """Generate a read-only SQL query for the application tracker from the question.

    The deterministic brain can't reason freely, so it maps the question to one of a few
    safe, parameterised shapes. A real model (OPENAI_API_KEY) writes the SQL directly.
    """
    q = question.lower()
    # Specific or colloquial statuses → a status filter.
    status_map = {
        "ghost": ["Ghosted"], "stale": ["Ghosted", "Applied"],
        "reject": ["Rejected"], "offer": ["Offer"],
        "interview": ["Interviewing"], "screen": ["Screening"],
        "pending": ["Applied", "Screening", "Interviewing"],
        "progress": ["Applied", "Screening", "Interviewing"],
        "waiting": ["Applied", "Screening", "Interviewing"],
        "respon": ["Screening", "Interviewing", "Offer"],  # "got a response"
    }
    for key, statuses in status_map.items():
        if key in q:
            vals = ", ".join(f"'{s}'" for s in statuses)
            return f"SELECT company, role, status, last_update FROM applications WHERE status IN ({vals})"
    # Default: the status rollup.
    return "SELECT status, COUNT(*) AS count FROM applications GROUP BY status"


def _called_tools(messages) -> set[str]:
    """Names of tools already invoked this turn (from the assistant's tool-call messages)."""
    names: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}).get("name") if isinstance(tc, dict) else None
            if fn:
                names.add(fn)
    return names


_SEARCH_WORDS = ("find", "search", "show", "roles", "role", "jobs", "job", "openings",
                 "opening", "hiring", "listings", "positions", "postings", "vacanc")
_COMPANY_CUES = ("tell me about", "about the company", "company ", "what is ", "what's ",
                 "info on", "roles at", "jobs at", "hiring at", "work at", "learn about")


def _parse_jobs_request(text: str) -> tuple[str, str, str, bool, bool]:
    """Pull (role, location, company, wants_search, wants_company) from a jobs request.

    Enables a multi-tool loop: 'find GenAI roles at Nimbus Labs' searches AND looks up
    the company; 'tell me about Nimbus Labs' only looks up the company."""
    low = text.lower()

    # Company name after a cue word (capitalized phrase, or words after "about"/"at").
    company = ""
    m = re.search(r"\b(?:about|at|on|for|is|like)\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,2})", text)
    if m:
        company = re.split(r"\b(and|that|which|hiring|recruiter)\b", m.group(1), flags=re.I)[0].strip(" .,")

    # Extract the role from the text with the "at <Company>" phrase removed, so the
    # company name doesn't leak into the job-board search term.
    role_text = re.sub(r"\b(?:at|about|for|on)\s+" + re.escape(company), " ", text, flags=re.I) if company else text
    role, location = _extract_job_query(role_text)
    wants_company = bool(company) and any(c in low for c in _COMPANY_CUES)
    wants_search = any(w in low for w in _SEARCH_WORDS)
    # A bare "tell me about X" with no search words is a company-only lookup.
    if wants_company and not wants_search:
        wants_search = False
    else:
        wants_search = wants_search or not wants_company
    return role, location, company, wants_search, wants_company


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
            # Keyword-match ONLY the current request. The router prepends the
            # conversation history ("Conversation so far:\n…\n\nCurrent request: …")
            # as context for a real LLM; matching the whole blob would let every past
            # "find jobs" / "draft email" turn keep re-triggering those agents.
            current = text.split("current request:", 1)[-1].strip() if "current request:" in text else text
            agents = [a for a, kws in _ROUTING_KEYWORDS.items() if any(k in current for k in kws)]
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
            subject, body = self._draft_message(user_prompt)
            return schema(channel="email", subject=subject, body=body)
        return schema.model_construct()

    @staticmethod
    def _draft_message(prompt: str) -> tuple[str, str]:
        """Write a real, tailored outreach draft instead of echoing the request.

        Parses the company, an optional role, and whether this is a cover letter or a
        follow-up, then fills a professional template. With OPENAI_API_KEY set, the real
        model writes this; offline we still produce something genuinely sendable.
        """
        text = prompt.strip()
        low = text.lower()

        # If the user is revising an existing draft, keep that body and just apply the
        # obvious tone asks; otherwise build fresh.
        if low.startswith("revise this draft"):
            body = text.split("Current body:", 1)[-1].strip()
            if "short" in low or "concise" in low:
                body = " ".join(body.split()[:60])
            return ("Following up on my application", body or "Thank you for your time and consideration.")

        # Company: "...at Acme", "to the recruiter at Acme", "for Acme".
        company = ""
        m = re.search(r"\b(?:at|with|to|for)\s+([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+){0,3})", text)
        if m:
            company = m.group(1).strip(" .,")
            # Trim trailing role words accidentally captured.
            company = re.split(r"\b(recruiter|team|about|regarding|for the)\b", company, flags=re.I)[0].strip()
        comp_phrase = f" at {company}" if company else ""

        # Role, if the request names one.
        role = ""
        rm = re.search(r"\b(?:for|as)\s+(?:a|an|the)?\s*([a-z][\w /+-]*?\b(?:engineer|developer|scientist|manager|lead|designer|analyst|architect))\b", low)
        if rm:
            role = rm.group(1).strip()
            role = re.sub(r"\bgenai\b", "GenAI", role.title(), flags=re.I)
        role_phrase = f" {role}" if role else ""

        is_cover = "cover letter" in low or "cover-letter" in low or "apply" in low

        if is_cover:
            subject = f"Application for the{role_phrase or ' role'}{comp_phrase}".strip()
            body = (
                f"Dear Hiring Team{comp_phrase},\n\n"
                f"I'm excited to apply for the{role_phrase or ' role'}{comp_phrase}. My background in "
                "building production software — and recent hands-on work with LLM applications, "
                "retrieval, and API development — maps closely to what the role calls for.\n\n"
                "I'd welcome the chance to share how I can contribute. My resume is attached, and I'm "
                "happy to walk through relevant projects at your convenience.\n\n"
                "Best regards,\n[Your name]"
            )
        else:
            subject = f"Following up on the{role_phrase or ''} application{comp_phrase}".replace("  ", " ").strip()
            body = (
                f"Hi{(' ' + company) if company else ' there'},\n\n"
                f"I recently applied for the{role_phrase or ' role'}{comp_phrase} and wanted to follow up to "
                "reaffirm my strong interest. I'd love to learn about the next steps in your process.\n\n"
                "If it's helpful, I'm glad to share more on my relevant experience or provide references. "
                "Thanks very much for your time and consideration.\n\n"
                "Best regards,\n[Your name]"
            )
        return subject, body

    async def get_tool_response(self, *, messages, tools) -> ToolResponse:
        tool_names = {t["function"]["name"] for t in tools} if tools else set()
        user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "") or ""
        called = _called_tools(messages)  # which tools have already run this turn

        if not tools:
            return ToolResponse(content=self._summarize(messages))

        # --- Jobs tools: a genuine multi-tool sequence ---------------------------
        # search_jobs for openings, then get_company to enrich when the user named a
        # company, so the trace shows multiple tools/calls like a real research loop.
        if {"search_jobs", "get_company"} & tool_names:
            role, location, company, wants_search, wants_company = _parse_jobs_request(user_text)
            if "search_jobs" in tool_names and wants_search and "search_jobs" not in called:
                return ToolResponse(
                    content=None,
                    tool_calls=[ToolCall("c1", "search_jobs", {"role": role, "location": location})],
                )
            if "get_company" in tool_names and company and wants_company and "get_company" not in called:
                return ToolResponse(
                    content=None,
                    tool_calls=[ToolCall("c2", "get_company", {"name": company})],
                )
            return ToolResponse(content=self._summarize(messages))

        if "run_sql_query" in tool_names:
            if "run_sql_query" in called:
                return ToolResponse(content=self._summarize(messages))
            return ToolResponse(content=None, tool_calls=[ToolCall("c1", "run_sql_query", {"query": _build_sql(user_text)})])
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
        # Lexical-grounding gate: the deterministic hashing embeddings give *every* doc a
        # small non-zero similarity, so cosine alone can't tell "weakly related" from
        # "unrelated". Only answer from a chunk that actually shares a content word with
        # the question; otherwise report not-found (correct groundedness + RBAC scope).
        q_words = {w for w in re.findall(r"[a-z]{3,}", question) if w not in _QUESTION_STOP}
        if q_words:
            for text in chunks.values():
                low = text.lower()
                if any(w in low for w in q_words):
                    return text
            return ""  # retrieved chunks exist but none is on-topic → not grounded
        return next(iter(chunks.values())) if chunks else ""

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
                name = p.get("name")
                if p.get("error"):
                    out.append(f"I couldn't find live info on {name} right now.")
                elif p.get("open_roles") is not None:  # live get_company (RemoteOK/Remotive)
                    line = f"{name} — {p['open_roles']} open role(s) live right now."
                    if p.get("sample_roles"):
                        line += " e.g. " + "; ".join(p["sample_roles"][:5]) + "."
                    out.append(line)
                elif p.get("summary"):  # offline fake profile
                    out.append(f"{name} — {p.get('industry', '')}, ~{p.get('size', '?')} people. {p['summary']}")
                else:
                    out.append(str(name))
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
