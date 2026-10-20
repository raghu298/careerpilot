"""Seed data for the offline demo: the user's resume, a target job description, and
career-advice knowledge documents.

Each doc carries an ``access_scope`` label so persona capability filtering is exercised
(``general`` is visible to everyone; ``premium`` only to the senior-switcher persona).
"""

from __future__ import annotations

KNOWLEDGE_DOCS: list[dict] = [
    {
        "source": "my-resume",
        "access_scope": "general",
        "content": (
            "Riya Sharma — Software Engineer, 5 years. Python, FastAPI, PostgreSQL, Docker, "
            "AWS. Built a retrieval-augmented support assistant (OpenAI + pgvector) and a "
            "multi-service payments backend. Led a 3-person team. Looking for Senior / GenAI "
            "backend roles. Gaps: limited Kubernetes in production, no formal LangGraph or "
            "agent-orchestration experience on the resume yet."
        ),
    },
    {
        "source": "jd-senior-genai-engineer",
        "access_scope": "general",
        "content": (
            "Job description — Senior GenAI Engineer. Required: Python, experience building "
            "LLM applications (RAG, tool-calling, agents), FastAPI, PostgreSQL/pgvector, and "
            "production deployment. Nice to have: LangGraph or similar orchestration, "
            "Kubernetes, evaluation/observability for LLM systems. Responsibilities: design "
            "multi-agent workflows, own retrieval quality, and ship reliable AI features."
        ),
    },
    {
        "source": "resume-tips",
        "access_scope": "general",
        "content": (
            "A strong resume bullet names the action, the technology, and a measurable result, "
            "e.g. 'Cut p95 latency 40% by adding a Redis read-through cache.' Tailor the top "
            "third of the resume to the specific job description, mirroring its key terms where "
            "they are genuinely true of your experience. One page per ~8 years."
        ),
    },
    {
        "source": "interview-behavioural",
        "access_scope": "general",
        "content": (
            "Answer behavioural questions with STAR: Situation, Task, Action, Result. Keep the "
            "Situation short, spend most of the answer on your specific Actions, and always end "
            "with a concrete Result and what you learned. Prepare 4-5 stories that can each flex "
            "to cover leadership, conflict, failure, and impact."
        ),
    },
    {
        "source": "salary-negotiation-premium",
        "access_scope": "premium",
        "content": (
            "Never give the first number if you can avoid it; ask for the role's budgeted range. "
            "Negotiate the whole package — base, bonus, equity, joining bonus, notice buyout — "
            "not just base. Anchor on market data and the value you bring, get the final offer in "
            "writing, and it is normal to ask for 2-3 days to decide."
        ),
    },
    {
        "source": "leadership-resume-premium",
        "access_scope": "premium",
        "content": (
            "For senior and lead roles, lead bullets with scope and outcome, not tasks: team "
            "size, systems owned, and business impact. Show evidence of multiplying others "
            "(mentoring, setting technical direction, cross-team influence) rather than only "
            "individual output."
        ),
    },
]
