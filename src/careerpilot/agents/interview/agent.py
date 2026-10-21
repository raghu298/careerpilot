"""Interview Prep agent — turns the user's resume + target JD into a focused mock
interview plan.

Pipeline mirrors the Knowledge agent: embed the request → retrieve scope-filtered
chunks from the vector store (the user's resume, the target job description, and any
interview-advice guides) → generate a schema-guarded ``InterviewPlan``. The agent then
renders the plan into readable, grouped markdown; the structured plan rides along in
``metadata`` for the UI / eval harness.

It is read-only and grounded: with no resume/JD in scope it still returns a sensible
generic plan but flags ``grounded=False`` so the caller knows it wasn't personalised.
"""

from __future__ import annotations

from collections import OrderedDict

from careerpilot.core.agent import AgentResult, Capabilities, OnEvent, emit
from careerpilot.core.embedding_client import EmbeddingClient
from careerpilot.core.llm_client import LLMClient
from careerpilot.core.tool_cache import ToolCache
from careerpilot.agents.interview.schemas import InterviewPlan
from careerpilot.integrations.vector_store.pgvector_store import VectorStore

_SYSTEM_PROMPT = """You are the interview-prep sub-agent of a job-hunt assistant.
From the provided context — the user's own resume, a target job description, and any
interview guides — produce a focused mock-interview plan: the questions this candidate
is most likely to be asked, grouped by category, each with a concrete talking point
drawn from THEIR resume and the JD (not generic advice).

Prefer questions the JD's requirements and the candidate's background actually imply.
Use the four categories: technical, behavioral, system-design, role-specific. Set
grounded=true only when you used the user's real resume/JD; if the context lacks them,
give a sensible generic plan and set grounded=false. Never invent experience the resume
does not show.
"""

_CATEGORY_TITLES = OrderedDict(
    [
        ("technical", "Technical"),
        ("system-design", "System design"),
        ("role-specific", "Role-specific"),
        ("behavioral", "Behavioral"),
    ]
)


class InterviewPrepAgent:
    name = "interview"

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        capabilities: Capabilities,
        *,
        k: int = 6,
    ) -> None:
        self._llm = llm_client
        self._embed = embedding_client
        self._store = vector_store
        self._caps = capabilities
        self._k = k

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult:
        if not self._caps.knowledge_labels:
            return AgentResult(self.name, "You don't have access to your career documents.")

        emit(on_event, {"type": "tool_called", "agent": self.name, "tool": "vector_search"})
        [query_vec] = await self._embed.embed([user_request])
        chunks = await self._store.search(
            embedding=query_vec, scopes=self._caps.knowledge_labels, k=self._k
        )
        emit(on_event, {"type": "tool_result", "agent": self.name, "tool": "vector_search", "count": len(chunks)})

        context = "\n\n".join(f"[{c.source}] {c.content}" for c in chunks)
        plan = await self._llm.get_structured_output(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Context:\n{context}\n\nRequest: {user_request}",
            schema=InterviewPlan,
        )
        return AgentResult(
            self.name,
            self._render(plan),
            metadata={
                "grounded": plan.grounded,
                "focus": plan.focus,
                "questions": [q.model_dump() for q in plan.questions],
            },
        )

    @staticmethod
    def _render(plan: InterviewPlan) -> str:
        if not plan.questions:
            return "I couldn't build an interview plan — add your resume and target JD via \"Use my resume\"."
        lines: list[str] = []
        if plan.focus:
            lines.append(f"**Focus:** {plan.focus}")
        if not plan.grounded:
            lines.append("_(generic plan — add your resume + target JD via \"Use my resume\" to tailor it)_")
        by_cat: dict[str, list] = {}
        for q in plan.questions:
            by_cat.setdefault(q.category, []).append(q)
        for cat, title in _CATEGORY_TITLES.items():
            group = by_cat.get(cat)
            if not group:
                continue
            lines.append(f"\n**{title}**")
            for q in group:
                lines.append(f"• {q.question}")
                if q.talking_point:
                    lines.append(f"  ↳ {q.talking_point}")
        return "\n".join(lines)
