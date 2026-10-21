"""In-process session store for the demo.

Each anonymous demo session owns a persona and a live LangGraph ``Orchestrator``
(which carries its own conversation memory, checkpointer, and any staged pending
action). Tokens are opaque random strings. This is deliberately in-memory: the public
demo is stateless across restarts, matching "inactive history expires" semantics.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from careerpilot.access_policy import resolve_capabilities
from careerpilot.bootstrap import SharedResources, build_session_orchestrator
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient
from careerpilot.orchestrator.orchestrator import Orchestrator

_SESSION_TTL_SECONDS = 24 * 60 * 60


@dataclass
class Session:
    token: str
    persona_id: str
    orchestrator: Orchestrator
    sql_client: SqliteSQLClient  # the session's real application tracker (SQLite)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, resources: SharedResources) -> None:
        self._resources = resources
        self._sessions: dict[str, Session] = {}

    def create(self, persona_id: str) -> Session:
        token = secrets.token_urlsafe(24)
        caps = resolve_capabilities(persona_id)
        sql_client = SqliteSQLClient(role=caps.sql_role)
        orch = build_session_orchestrator(
            self._resources, persona_id=persona_id, sql_client=sql_client
        )
        session = Session(
            token=token, persona_id=persona_id, orchestrator=orch, sql_client=sql_client
        )
        self._sessions[token] = session
        return session

    def get(self, token: str) -> Session | None:
        session = self._sessions.get(token)
        if session is None:
            return None
        if time.time() - session.last_seen > _SESSION_TTL_SECONDS:
            self._sessions.pop(token, None)
            return None
        session.last_seen = time.time()
        return session
