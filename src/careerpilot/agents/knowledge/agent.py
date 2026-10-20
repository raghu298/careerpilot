"""Knowledge Agent — retrieval-augmented generation over financial-literacy docs.

Pipeline: embed the query → retrieve scope-filtered chunks from the vector store →
generate a schema-guarded answer. ``retrieval_floor`` is a cheap "is there any signal
at all" check, not a per-chunk relevance filter; once anything clears it, every
retrieved chunk goes to the model and ``answer_found`` is the real groundedness gate.
"""

from __future__ import annotations

from careerpilot.core.agent import AgentResult, Capabilities, OnEvent, emit
from careerpilot.core.embedding_client import EmbeddingClient
from careerpilot.core.llm_client import LLMClient
from careerpilot.core.tool_cache import ToolCache
from careerpilot.agents.knowledge.schemas import KnowledgeAnswer
from careerpilot.integrations.vector_store.pgvector_store import VectorStore

_SYSTEM_PROMPT = """You are the knowledge sub-agent of a job-hunt / career assistant.
Answer using ONLY the provided context passages — the user's own resume, a target job
description, and career-advice guides. For "do I fit this role / what are my gaps"
questions, compare the resume against the job description and be specific about matches
and gaps. If the context does not contain the answer, set answer_found to false and say
so — never invent experience, requirements, or advice. Cite the sources you used.
"""


class KnowledgeAgent:
    name = "knowledge"

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        capabilities: Capabilities,
        *,
        k: int = 5,
        retrieval_floor: float = 0.2,
    ) -> None:
        self._llm = llm_client
        self._embed = embedding_client
        self._store = vector_store
        self._caps = capabilities
        self._k = k
        self._retrieval_floor = retrieval_floor

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult:
        if not self._caps.knowledge_labels:
            return AgentResult(self.name, "You don't have access to the knowledge base.")

        emit(on_event, {"type": "tool_called", "agent": self.name, "tool": "vector_search"})
        [query_vec] = await self._embed.embed([user_request])
        chunks = await self._store.search(
            embedding=query_vec, scopes=self._caps.knowledge_labels, k=self._k
        )
        emit(on_event, {"type": "tool_result", "agent": self.name, "tool": "vector_search", "count": len(chunks)})

        if not chunks or chunks[0].score < self._retrieval_floor:
            return AgentResult(
                self.name,
                "I couldn't find anything about that in your career documents.",
                metadata={"answer_found": False, "citations": []},
            )

        context = "\n\n".join(f"[{c.source}] {c.content}" for c in chunks)
        answer = await self._llm.get_structured_output(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Context:\n{context}\n\nQuestion: {user_request}",
            schema=KnowledgeAnswer,
        )
        return AgentResult(
            self.name,
            answer.answer,
            metadata={"answer_found": answer.answer_found, "citations": answer.citations},
        )
