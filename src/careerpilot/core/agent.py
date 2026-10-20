"""Core agent contracts.

Every specialist agent (knowledge, analytics, market, notify) implements the same
small `Agent` protocol so the orchestrator can treat them identically. Agents that
produce a side effect (the Notify agent) additionally implement `ConfirmableAgent`,
checked at runtime so the read-only agents never carry no-op confirm methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from careerpilot.core.tool_cache import ToolCache

# A live-trace callback. The orchestrator and agents fire these as routing and tool
# calls actually happen, so the API layer can stream them to the browser over SSE.
# Deliberately plain dicts: the shape varies per event ``type`` and this is an
# observability concern, not a guardrail boundary that needs schema validation.
OnEvent = Callable[[dict], None]


def emit(on_event: OnEvent | None, event: dict) -> None:
    """Fire ``event`` if a callback was supplied; otherwise do nothing.

    Lets every call site emit unconditionally instead of guarding with ``if on_event``.
    """
    if on_event is not None:
        on_event(event)


@dataclass
class AgentResult:
    """Uniform result every agent hands back to the orchestrator."""

    agent_name: str
    content: str
    metadata: dict | None = None


class Agent(Protocol):
    """The single interface the orchestrator depends on.

    ``history`` is prior turns in OpenAI chat-message shape (``None`` on a first turn).
    ``on_event`` is the live-trace callback. ``tool_cache`` is a session-scoped read
    cache used only by agents whose tool calls are cacheable external reads. All three
    are optional so simple callers and tests stay trivial.
    """

    name: str

    async def handle(
        self,
        user_request: str,
        history: list[dict] | None = None,
        on_event: OnEvent | None = None,
        tool_cache: ToolCache | None = None,
    ) -> AgentResult: ...


@dataclass
class PendingAction:
    """A staged, not-yet-executed side effect awaiting human confirmation."""

    kind: str  # e.g. "alert" | "reminder"
    destination: str  # resolved by policy, never by the LLM
    subject: str
    body: str
    preview: str


@runtime_checkable
class ConfirmableAgent(Protocol):
    """Extra interface for an agent that stages an action before executing it.

    Checked with ``isinstance(agent, ConfirmableAgent)`` so read-only agents need no
    changes and the orchestrator stays generic about which agent this applies to.
    """

    def has_pending(self) -> bool: ...
    def pending(self) -> PendingAction | None: ...
    async def confirm_pending(self) -> AgentResult: ...
    def cancel_pending(self) -> AgentResult: ...
    async def revise_pending(self, edit_instructions: str) -> AgentResult: ...


@dataclass
class Capabilities:
    """The typed capability set a persona is resolved into before any agent is built.

    Access is decided here, not left to prompt text. Each agent only receives the
    surfaces its persona is allowed to touch.
    """

    persona_id: str
    display_name: str
    knowledge_labels: tuple[str, ...] = field(default_factory=tuple)
    sql_role: str = "none"  # "none" | "basic" | "full"
    can_use_jobs: bool = True
    can_stage_notifications: bool = True
