"""Conversation memory lives on the orchestrator, not per-agent.

Routing can send consecutive turns to different agents, so a per-agent history would
lose continuity the moment that happens. This is a rolling FIFO window of the last N
turns, flattened into the router prompt and threaded into each agent as ``history``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self._max_turns = max_turns
        self._turns: list[Turn] = []

    def add_user(self, content: str) -> None:
        self._append(Turn("user", content))

    def add_assistant(self, content: str) -> None:
        self._append(Turn("assistant", content))

    def _append(self, turn: Turn) -> None:
        self._turns.append(turn)
        # Keep the last max_turns exchanges (a user+assistant pair counts as two).
        overflow = len(self._turns) - self._max_turns * 2
        if overflow > 0:
            self._turns = self._turns[overflow:]

    def as_messages(self) -> list[dict]:
        """OpenAI chat-message shape, for threading into an agent's tool loop."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def as_text(self) -> str:
        """Flattened transcript, for the router's follow-up disambiguation."""
        return "\n".join(f"{t.role}: {t.content}" for t in self._turns)

    def seed(self, turns: list[Turn]) -> None:
        """Rehydrate from durable storage (API path)."""
        self._turns = list(turns)[-self._max_turns * 2 :]
