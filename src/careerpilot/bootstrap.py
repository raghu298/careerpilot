"""Shared wiring reused by every entry point (CLI, API, tests).

``build_shared_resources`` builds the expensive, stateless pieces once: the LLM and
embedding clients (real OpenAI when ``OPENAI_API_KEY`` is set, deterministic offline
stand-ins otherwise), the router, the seeded vector store, the market source, and the
SQL client. ``build_session_orchestrator`` is the cheap per-session slice: it resolves
a persona into capabilities, constructs the four agents and a fresh Notify agent with
its own Demo Outbox, and returns a LangGraph ``Orchestrator``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from careerpilot.access_policy import resolve_capabilities
from careerpilot.agents.analytics.agent import AnalyticsAgent
from careerpilot.agents.knowledge.agent import KnowledgeAgent
from careerpilot.agents.jobs.agent import JobsAgent
from careerpilot.agents.notify.agent import NotifyAgent
from careerpilot.core.chunking import chunk_document
from careerpilot.core.embedding_client import EmbeddingClient, OpenAIEmbeddingClient
from careerpilot.core.llm_client import LLMClient, OpenAILLMClient
from careerpilot.fixtures import KNOWLEDGE_DOCS
import asyncio
from pathlib import Path

from careerpilot.integrations.jobs_data.mcp_client import FakeJobsMCP, JobsSource, MCPJobsSource
from careerpilot.integrations.notifications.notifiers import DemoOutbox
from careerpilot.integrations.offline.offline_llm import HashingEmbeddingClient, OfflineLLMClient
from careerpilot.integrations.sql_db.fake_sql import FakeSQLClient
from careerpilot.integrations.vector_store.pgvector_store import InMemoryVectorStore, VectorStore
from careerpilot.orchestrator.memory import ConversationMemory
from careerpilot.orchestrator.orchestrator import Orchestrator
from careerpilot.orchestrator.router import Router


@dataclass
class SharedResources:
    llm: LLMClient
    embeddings: EmbeddingClient
    router: Router
    vector_store: VectorStore
    jobs_source: JobsSource
    offline: bool
    jobs_mode: str = "fake"  # "live-mcp" | "remote-mcp" | "fake"
    mcp_aclose: object | None = None  # async callable to tear down the MCP session


def _real_mode() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


async def seed_knowledge_store(embeddings: EmbeddingClient, docs: list[dict]) -> InMemoryVectorStore:
    """Chunk + embed a set of documents into a fresh vector store. Reused for the shared
    demo corpus and for a per-session store when a user supplies their own resume/JD."""
    store = InMemoryVectorStore()
    for doc in docs:
        for chunk in chunk_document(doc["content"]):
            [vec] = await embeddings.embed([chunk])
            await store.add(
                content=chunk, embedding=vec, source=doc["source"], access_scope=doc["access_scope"]
            )
    return store


async def build_shared_resources() -> SharedResources:
    real = _real_mode()
    llm: LLMClient = OpenAILLMClient() if real else OfflineLLMClient()
    embeddings: EmbeddingClient = OpenAIEmbeddingClient() if real else HashingEmbeddingClient()

    vector_store = await seed_knowledge_store(embeddings, KNOWLEDGE_DOCS)

    jobs_source, jobs_mode, mcp_aclose = await _build_jobs_source()

    return SharedResources(
        llm=llm,
        embeddings=embeddings,
        router=Router(llm),
        vector_store=vector_store,
        jobs_source=jobs_source,
        offline=not real,
        jobs_mode=jobs_mode,
        mcp_aclose=mcp_aclose,
    )


async def _build_jobs_source() -> tuple[JobsSource, str, object | None]:
    """Prefer a real MCP source; always fall back to the fake so the app still boots.

    - ``JOBS_MCP_URL`` set  → connect to that remote MCP server (streamable HTTP).
    - otherwise             → launch the bundled live jobs MCP server over stdio
                              (real MCP round-trip, live Remotive data).
    - any failure           → deterministic FakeJobsMCP.
    """
    url = os.environ.get("JOBS_MCP_URL")
    try:
        if url:
            source: JobsSource = MCPJobsSource(url=url, token=os.environ.get("JOBS_MCP_TOKEN"))
            mode = "remote-mcp"
        else:
            script = Path(__file__).resolve().parents[2] / "mcp_server" / "jobs_mcp_server.py"
            if not script.exists():
                return FakeJobsMCP(), "fake", None
            source = MCPJobsSource(script_path=str(script))
            mode = "live-mcp"
        # Probe once (bounded) so we fall back to the fake if the server or its live
        # data source isn't reachable — the app must always boot and answer.
        await asyncio.wait_for(source.call_tool("search_jobs", {"role": "engineer"}), timeout=15)
        return source, mode, None
    except Exception:
        return FakeJobsMCP(), "fake", None


def build_session_orchestrator(
    resources: SharedResources, *, persona_id: str, memory: ConversationMemory | None = None
) -> Orchestrator:
    caps = resolve_capabilities(persona_id)
    demo_outbox = DemoOutbox()
    # The deterministic offline embeddings produce lower cosine scores than real
    # embeddings, so a valid match can fall just under the default 0.2 floor. Relax the
    # floor in offline mode; real embeddings (OPENAI_API_KEY) keep the stricter 0.2.
    retrieval_floor = 0.05 if resources.offline else 0.2
    agents = {
        "knowledge": KnowledgeAgent(
            resources.llm, resources.embeddings, resources.vector_store, caps, retrieval_floor=retrieval_floor
        ),
        "jobs": JobsAgent(resources.llm, resources.jobs_source, caps),
        "analytics": AnalyticsAgent(resources.llm, FakeSQLClient(role=caps.sql_role), caps),
        "notify": NotifyAgent(resources.llm, caps, demo_outbox=demo_outbox),
    }
    orch = Orchestrator(resources.router, agents, memory=memory)
    # Expose the session outbox so the API's /outbox endpoint can read captures.
    orch.demo_outbox = demo_outbox  # type: ignore[attr-defined]
    return orch
