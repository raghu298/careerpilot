"""Session store for the demo, with durable persistence.

Each anonymous demo session owns a persona and a live LangGraph ``Orchestrator`` (which
carries its own conversation memory, checkpointer, and any staged pending action). Live
orchestrators stay in memory for speed, but each session's persona, application tracker,
and conversation memory are mirrored to a ``PersistenceStore`` so a server restart /
Render cold-start rehydrates the session instead of 401-ing the visitor. (Pending HITL
drafts are the one thing not restored across a restart — they live only in the paused
in-memory graph.)
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from careerpilot.access_policy import resolve_capabilities
from careerpilot.bootstrap import SharedResources, build_session_orchestrator
from careerpilot.integrations.sql_db.sqlite_client import SqliteSQLClient
from careerpilot.orchestrator.memory import ConversationMemory, Turn
from careerpilot.orchestrator.orchestrator import Orchestrator

from api.persistence import PersistenceStore

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
    def __init__(self, resources: SharedResources, persistence: PersistenceStore | None = None) -> None:
        self._resources = resources
        self._sessions: dict[str, Session] = {}
        self._store = persistence

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
        if self._store is not None:
            self._store.save_session(token, persona_id, session.created_at, session.last_seen)
            self._store.save_applications(token, sql_client.all_rows())  # starter rows
        return session

    def get(self, token: str) -> Session | None:
        session = self._sessions.get(token)
        if session is None:
            session = self._rehydrate(token)
            if session is None:
                return None
        if time.time() - session.last_seen > _SESSION_TTL_SECONDS:
            self._sessions.pop(token, None)
            if self._store is not None:
                self._store.delete_session(token)
            return None
        session.last_seen = time.time()
        if self._store is not None:
            self._store.touch_session(token, session.last_seen)
        return session

    def _rehydrate(self, token: str) -> Session | None:
        """Rebuild a session from durable storage after a restart."""
        if self._store is None:
            return None
        row = self._store.load_session(token)
        if row is None:
            return None
        if time.time() - row["last_seen"] > _SESSION_TTL_SECONDS:
            self._store.delete_session(token)
            return None
        persona_id = row["persona_id"]
        caps = resolve_capabilities(persona_id)
        sql_client = SqliteSQLClient(role=caps.sql_role, seed=False)
        apps = self._store.load_applications(token)
        if apps:
            sql_client.replace_applications(apps)
        memory = ConversationMemory()
        memory.seed([Turn(m["role"], m["content"]) for m in self._store.load_memory(token)])
        orch = build_session_orchestrator(
            self._resources, persona_id=persona_id, memory=memory, sql_client=sql_client
        )
        session = Session(
            token=token, persona_id=persona_id, orchestrator=orch, sql_client=sql_client,
            created_at=row["created_at"], last_seen=row["last_seen"],
        )
        self._sessions[token] = session
        return session

    # ---- persistence hooks called by the API after a mutation ----
    def persist_applications(self, session: Session) -> None:
        if self._store is not None:
            self._store.save_applications(session.token, session.sql_client.all_rows())

    def persist_memory(self, session: Session) -> None:
        if self._store is not None:
            self._store.save_memory(session.token, session.orchestrator.memory.as_messages())
