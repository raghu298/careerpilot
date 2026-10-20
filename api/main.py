"""FastAPI backend for CareerPilot.

Endpoints mirror the NexusAI demo shape:

    GET  /health                 → liveness
    GET  /demo/personas          → the selectable personas
    POST /demo/sessions          → create an anonymous persona session (returns a token)
    POST /chat                   → run a turn, streamed as Server-Sent Events (live trace)
    POST /pending                → resolve a staged action (confirm / cancel / revise)
    GET  /demo/outbox            → this session's captured notifications

The chat route turns the orchestrator's live-trace callbacks into an SSE stream that
the frontend's Orchestration Trace panel renders as agents coordinate.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from careerpilot.access_policy import list_personas
from careerpilot.bootstrap import build_shared_resources
from api.sessions import Session, SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = await build_shared_resources()
    app.state.sessions = SessionStore(app.state.resources)
    yield


def _allowed_origins() -> list[str]:
    """Lock CORS to FRONTEND_ORIGIN when it is set (plus localhost for dev); fall back
    to open access only when it is unset or explicitly "*", so a fresh deploy still
    works out of the box but can be tightened with a single env var."""
    configured = os.environ.get("FRONTEND_ORIGIN", "").strip()
    if configured and configured != "*":
        return [configured, "http://localhost:3000"]
    return ["*"]


app = FastAPI(title="CareerPilot API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionRequest(BaseModel):
    persona_id: str


class ChatRequest(BaseModel):
    session: str
    message: str


class PendingRequest(BaseModel):
    session: str
    action: str  # "confirm" | "cancel" | "revise"
    edit: str | None = None


def _require_session(request_token: str) -> Session:
    store: SessionStore = app.state.sessions
    session = store.get(request_token)
    if session is None:
        raise HTTPException(status_code=401, detail="Unknown or expired session")
    return session


@app.get("/health")
async def health():
    res = getattr(app.state, "resources", None)
    return {
        "status": "ok",
        "offline": getattr(res, "offline", None),
        "jobs_mode": getattr(res, "jobs_mode", None),
    }


@app.get("/demo/personas")
async def personas():
    return {
        "personas": [
            {"id": c.persona_id, "name": c.display_name, "sql_role": c.sql_role}
            for c in list_personas()
        ]
    }


@app.post("/demo/sessions")
async def create_session(req: SessionRequest):
    try:
        session = app.state.sessions.create(req.persona_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Unknown persona")
    return {"session": session.token, "persona_id": session.persona_id}


def _result_payload(out) -> dict:
    return {
        "type": "result",
        "routing_reasoning": out.routing_reasoning,
        "replies": [
            {"agent": name, "content": r.content, "metadata": r.metadata or {}}
            for name, r in out.results.items()
        ],
        "pending": None
        if out.pending is None
        else {
            "agent": out.pending_agent,
            "kind": out.pending.kind,
            "destination": out.pending.destination,
            "subject": out.pending.subject,
            "body": out.pending.body,
            "preview": out.pending.preview,
        },
    }


async def _stream(coro_factory):
    """Run an orchestrator coroutine while forwarding its trace events as SSE."""
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def on_event(event: dict) -> None:
        queue.put_nowait(event)

    async def runner():
        try:
            out = await coro_factory(on_event)
            queue.put_nowait(_result_payload(out))
        except Exception as exc:  # surface errors to the client as a trace event
            queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            queue.put_nowait(done)

    task = asyncio.create_task(runner())
    try:
        while True:
            event = await queue.get()
            if event is done:
                break
            yield {"data": json.dumps(event)}
    finally:
        await task


@app.post("/chat")
async def chat(req: ChatRequest):
    session = _require_session(req.session)

    async def factory(on_event):
        return await session.orchestrator.handle(req.message, on_event=on_event)

    return EventSourceResponse(_stream(factory))


@app.post("/pending")
async def pending(req: PendingRequest):
    session = _require_session(req.session)
    decision = {"action": req.action, "edit": req.edit or ""}

    async def factory(on_event):
        return await session.orchestrator.resume_pending(decision, on_event=on_event)

    return EventSourceResponse(_stream(factory))


@app.get("/demo/outbox")
async def outbox(x_session: str = Header(default="")):
    session = _require_session(x_session)
    box = getattr(session.orchestrator, "demo_outbox", None)
    items = [] if box is None else [
        {"channel": d.channel, "destination": d.destination, "subject": d.subject, "body": d.body}
        for d in box.items()
    ]
    return {"items": items}


@app.exception_handler(HTTPException)
async def http_exc_handler(_request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
