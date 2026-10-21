"""The traffic cop, built on LangGraph.

Flow as a ``StateGraph``:

    START → router → (Send fan-out) → run_agent × N → aggregate → END

- ``router`` calls the schema-guarded ``Router`` and chooses the agents.
- A conditional edge fans out to one ``run_agent`` invocation per selected agent
  using LangGraph's ``Send`` API, so they run concurrently; a reducer merges each
  agent's result back into shared state.
- ``aggregate`` detects a staged side effect from a ``ConfirmableAgent`` and pauses
  the graph with ``interrupt()`` (human-in-the-loop). The caller resumes with a
  Send/Edit/Cancel decision via ``resume_pending``.

The graph depends only on the ``Agent`` protocol and a name→agent dict — it has no
knowledge of concrete agent classes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated
from uuid import uuid4

from typing_extensions import NotRequired, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from careerpilot.core.agent import (
    Agent,
    AgentResult,
    ConfirmableAgent,
    OnEvent,
    PendingAction,
    emit,
)
from careerpilot.core.observability import metrics
from careerpilot.core.safety import content_check
from careerpilot.core.tool_cache import ToolCache
from careerpilot.orchestrator.memory import ConversationMemory
from careerpilot.orchestrator.router import Router


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


_CAPABILITIES = (
    "I'm CareerPilot, your job-hunt copilot. Ask me and I'll route you to the right specialist:\n\n"
    "• **Résumé & JD fit** — \"Does my resume fit this role? What are my gaps?\" (upload yours via "
    "\"Use my resume\")\n"
    "• **Live job search** — \"Find senior GenAI roles\" — real postings with apply links\n"
    "• **Your application tracker** — \"How many of my applications got a response? Which are stale?\" "
    "(add yours via \"My applications\")\n"
    "• **Outreach drafts** — \"Draft a follow-up email to the recruiter at Acme\" — I draft, you confirm before it's sent\n\n"
    "What would you like to start with?"
)

_GREETINGS = {"hi", "hey", "hello", "yo", "sup", "hiya", "howdy", "thanks", "thank you", "ok", "okay"}


def _help_reply(text: str) -> str | None:
    """Return a capabilities/greeting reply for meta messages, else None (route normally)."""
    t = text.strip().lower().rstrip("!?. ")
    if t in _GREETINGS:
        return _CAPABILITIES
    meta_phrases = (
        "what can you do", "what all you can do", "what do you do", "what can you help",
        "how can you help", "help me", "what are you", "who are you", "how does this work",
        "what is this", "capabilities", "how to use", "what should i ask",
    )
    if any(p in t for p in meta_phrases):
        return _CAPABILITIES
    return None


class GraphState(TypedDict):
    user_request: str
    history: list[dict]
    selected: NotRequired[list[str]]
    routing_reasoning: NotRequired[str]
    results: Annotated[dict[str, dict], _merge]
    pending: NotRequired[dict | None]
    pending_agent: NotRequired[str | None]
    # populated per-fan-out invocation via Send:
    agent_name: NotRequired[str]


@dataclass
class OrchestratorResult:
    results: dict[str, AgentResult] = field(default_factory=dict)
    routing_reasoning: str = ""
    pending: PendingAction | None = None
    pending_agent: str | None = None

    @property
    def combined_text(self) -> str:
        return "\n\n".join(r.content for r in self.results.values() if r.content)


class Orchestrator:
    def __init__(
        self,
        router: Router,
        agents: dict[str, Agent],
        *,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._router = router
        self._agents = agents
        self._memory = memory or ConversationMemory()
        self._checkpointer = MemorySaver()
        self._graph = self._build_graph().compile(checkpointer=self._checkpointer)
        self._thread_id = uuid4().hex  # one checkpoint thread per orchestrator (per session)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    # ---- graph construction -------------------------------------------------

    def _build_graph(self) -> StateGraph:
        g = StateGraph(GraphState)
        g.add_node("router", self._router_node)
        g.add_node("run_agent", self._run_agent_node)
        g.add_node("aggregate", self._aggregate_node)
        g.add_edge(START, "router")
        g.add_conditional_edges("router", self._fan_out, ["run_agent"])
        g.add_edge("run_agent", "aggregate")
        g.add_edge("aggregate", END)
        return g

    async def _router_node(self, state: GraphState, config) -> dict:
        on_event = self._on_event(config)
        decision = await self._router.route(state["user_request"], self._memory.as_text())
        selected = [n for n in decision.agents if n in self._agents]
        emit(on_event, {"type": "routing_decided", "agents": selected, "reasoning": decision.reasoning})
        return {"selected": selected, "routing_reasoning": decision.reasoning, "results": {}}

    def _fan_out(self, state: GraphState) -> list[Send]:
        return [
            Send(
                "run_agent",
                {
                    "agent_name": name,
                    "user_request": state["user_request"],
                    "history": state["history"],
                    "results": {},
                },
            )
            for name in state.get("selected", [])
        ]

    async def _run_agent_node(self, state: GraphState, config) -> dict:
        name = state["agent_name"]
        on_event = self._on_event(config)
        tool_cache = self._tool_cache(config)
        emit(on_event, {"type": "agent_started", "agent": name})
        metrics.incr(f"agent.{name}.calls")
        result = await self._agents[name].handle(
            state["user_request"],
            history=state.get("history"),
            on_event=on_event,
            tool_cache=tool_cache,
        )
        emit(on_event, {"type": "agent_finished", "agent": name})
        return {"results": {name: _result_to_dict(result)}}

    async def _aggregate_node(self, state: GraphState, config) -> dict:
        on_event = self._on_event(config)
        for name in state.get("selected", []):
            agent = self._agents[name]
            if isinstance(agent, ConfirmableAgent) and agent.has_pending():
                pending = agent.pending()
                emit(on_event, {"type": "pending_action", "agent": name, "kind": pending.kind})
                # Pause the graph for a human decision (Send / Edit / Cancel).
                decision = interrupt(
                    {"agent": name, "pending": _pending_to_dict(pending)}
                )
                resolved = await self._apply_decision(agent, decision)
                return {"results": {name: _result_to_dict(resolved)}, "pending": None}
        return {}

    async def _apply_decision(self, agent: ConfirmableAgent, decision: dict) -> AgentResult:
        action = (decision or {}).get("action", "cancel")
        if action == "confirm":
            return await agent.confirm_pending()
        if action == "revise":
            return await agent.revise_pending(decision.get("edit", ""))
        return agent.cancel_pending()

    # ---- public API ---------------------------------------------------------

    async def handle(self, user_request: str, on_event: OnEvent | None = None) -> OrchestratorResult:
        metrics.incr("orchestrator.requests")

        # Guardrail: block clearly out-of-scope / unsafe requests with a safe redirect.
        safe = content_check(user_request)
        if safe is not None:
            metrics.incr("safety.blocked")
            emit(on_event, {"type": "routing_decided", "agents": [], "reasoning": "Blocked by safety guardrail."})
            emit(on_event, {"type": "done"})
            return OrchestratorResult(
                results={"system": AgentResult("system", safe)},
                routing_reasoning="Blocked by safety guardrail.",
            )

        # Meta / greeting / off-topic messages ("what can you do", "hi") shouldn't be
        # routed into RAG and answered with a random document — answer them directly.
        help_text = _help_reply(user_request)
        if help_text is not None:
            metrics.incr("orchestrator.help_reply")
            emit(on_event, {"type": "routing_decided", "agents": [], "reasoning": "Capabilities / greeting — answered directly."})
            self._memory.add_user(user_request)
            self._memory.add_assistant(help_text)
            emit(on_event, {"type": "done"})
            return OrchestratorResult(
                results={"assistant": AgentResult("assistant", help_text)},
                routing_reasoning="Capabilities / greeting — answered directly.",
            )

        config = self._config(on_event)
        try:
            with metrics.timer("orchestrator.latency_ms"):
                state = await self._graph.ainvoke(
                    {"user_request": user_request, "history": self._memory.as_messages(), "results": {}},
                    config=config,
                )
        except Exception:
            metrics.incr("orchestrator.errors")
            raise
        return self._finalize(user_request, state, on_event)

    async def resume_pending(self, decision: dict, on_event: OnEvent | None = None) -> OrchestratorResult:
        """Resume a graph paused at the HITL interrupt with a Send/Edit/Cancel decision."""
        config = self._config(on_event)
        state = await self._graph.ainvoke(Command(resume=decision), config=config)
        return self._finalize(None, state, on_event)

    # ---- helpers ------------------------------------------------------------

    def _finalize(self, user_request: str | None, state: dict, on_event: OnEvent | None) -> OrchestratorResult:
        interrupts = state.get("__interrupt__")
        # The graph's `results` channel is checkpointed per session thread and merged by a
        # reducer, so it accumulates every agent that has ever run in this session. Return
        # only THIS turn's agents (state["selected"]) so a later turn doesn't re-emit prior
        # turns' replies.
        selected = set(state.get("selected", []))
        results = {
            n: _dict_to_result(d)
            for n, d in state.get("results", {}).items()
            if not selected or n in selected
        }
        out = OrchestratorResult(results=results, routing_reasoning=state.get("routing_reasoning", ""))
        if interrupts:
            payload = interrupts[0].value
            out.pending_agent = payload.get("agent")
            out.pending = _dict_to_pending(payload.get("pending", {}))
        else:
            if user_request is not None:
                self._memory.add_user(user_request)
            if out.combined_text:
                self._memory.add_assistant(out.combined_text)
            emit(on_event, {"type": "done"})
        return out

    def _config(self, on_event: OnEvent | None) -> dict:
        return {
            "configurable": {
                "thread_id": self._thread_id,
                "on_event": on_event,
                "tool_cache": ToolCache(),
            }
        }

    @staticmethod
    def _on_event(config) -> OnEvent | None:
        return (config or {}).get("configurable", {}).get("on_event")

    @staticmethod
    def _tool_cache(config) -> ToolCache | None:
        return (config or {}).get("configurable", {}).get("tool_cache")


# ---- (de)serialization between dataclasses and graph state ------------------


def _result_to_dict(r: AgentResult) -> dict:
    return {"agent_name": r.agent_name, "content": r.content, "metadata": r.metadata}


def _dict_to_result(d: dict) -> AgentResult:
    return AgentResult(agent_name=d["agent_name"], content=d["content"], metadata=d.get("metadata"))


def _pending_to_dict(p: PendingAction) -> dict:
    return {
        "kind": p.kind,
        "destination": p.destination,
        "subject": p.subject,
        "body": p.body,
        "preview": p.preview,
    }


def _dict_to_pending(d: dict) -> PendingAction | None:
    if not d:
        return None
    return PendingAction(
        kind=d["kind"],
        destination=d["destination"],
        subject=d["subject"],
        body=d["body"],
        preview=d["preview"],
    )


# Keep a module-level handle on asyncio for callers that want to run outside an
# existing loop (e.g. the CLI script).
__all__ = ["Orchestrator", "OrchestratorResult", "asyncio"]
