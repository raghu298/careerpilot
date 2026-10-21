"""FastAPI backend for CareerPilot.

Endpoints mirror the NexusAI demo shape:

    GET  /health                 → liveness
    GET  /metrics                → observability snapshot (counters + latency)
    POST /auth/signup|login      → optional accounts (hashed pw → signed bearer token)
    GET  /auth/me                → verify the current bearer token
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

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from careerpilot.access_policy import list_personas
from careerpilot.bootstrap import build_session_orchestrator, build_shared_resources, seed_knowledge_store
from careerpilot.core.observability import get_logger, metrics
from careerpilot.core.rate_limit import RateLimiter
from careerpilot.core.safety import redact_pii
from careerpilot.fixtures import KNOWLEDGE_DOCS
from api.auth import bearer_email, hash_password, make_token, require_auth_enabled, verify_password
from api.persistence import PersistenceStore
from api.sessions import Session, SessionStore

logger = get_logger("api")

# Rate limiters (token buckets). Chat: ~1 req/s sustained, burst 10, per session.
# Session creation: slower, per client IP, to blunt token farming.
_chat_limiter = RateLimiter(rate_per_sec=1.0, capacity=10)
_session_limiter = RateLimiter(rate_per_sec=0.2, capacity=8)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = await build_shared_resources()
    app.state.persistence = PersistenceStore()
    app.state.persistence.purge_expired(24 * 60 * 60)
    app.state.sessions = SessionStore(app.state.resources, app.state.persistence)
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


class DocsRequest(BaseModel):
    session: str
    resume: str = ""
    job_description: str = ""


class Application(BaseModel):
    company: str
    role: str
    status: str = "Applied"
    applied_on: str = ""
    last_update: str = ""
    source: str = ""
    location: str = ""


class ApplicationsRequest(BaseModel):
    session: str
    applications: list[Application]


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


class AuthRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/signup")
async def signup(req: AuthRequest):
    email = req.email.strip().lower()
    if "@" not in email or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Valid email and a 6+ char password required.")
    salt, pw_hash = hash_password(req.password)
    if not app.state.persistence.create_user(email, salt, pw_hash):
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    metrics.incr("auth.signups")
    return {"token": make_token(email), "email": email}


@app.post("/auth/login")
async def login(req: AuthRequest):
    email = req.email.strip().lower()
    user = app.state.persistence.get_user(email)
    if user is None or not verify_password(req.password, user["salt"], user["pw_hash"]):
        metrics.incr("auth.login_failures")
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    metrics.incr("auth.logins")
    return {"token": make_token(email), "email": email}


@app.get("/auth/me")
async def auth_me(authorization: str = Header(default="")):
    email = bearer_email(authorization)
    if email is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"email": email}


@app.get("/metrics")
async def get_metrics():
    """Observability snapshot — counters + latency percentiles for routing, agents,
    LLM/embedding calls, rate-limiting and safety. In-process; resets on restart."""
    return metrics.snapshot()


@app.get("/demo/personas")
async def personas():
    return {
        "personas": [
            {"id": c.persona_id, "name": c.display_name, "sql_role": c.sql_role}
            for c in list_personas()
        ]
    }


@app.post("/demo/sessions")
async def create_session(req: SessionRequest, request: Request, authorization: str = Header(default="")):
    # Auth is optional; enforced only when CP_REQUIRE_AUTH is set.
    if require_auth_enabled() and bearer_email(authorization) is None:
        raise HTTPException(status_code=401, detail="Sign in to start a session.")
    ip = _client_ip(request)
    if not _session_limiter.allow(ip):
        metrics.incr("ratelimit.sessions.blocked")
        raise HTTPException(status_code=429, detail="Too many sessions — slow down a moment.")
    try:
        session = app.state.sessions.create(req.persona_id)
    except KeyError:
        raise HTTPException(status_code=400, detail="Unknown persona")
    metrics.incr("sessions.created")
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
    if not _chat_limiter.allow(req.session):
        metrics.incr("ratelimit.chat.blocked")
        retry = _chat_limiter.retry_after(req.session)
        raise HTTPException(status_code=429, detail=f"Too many messages — retry in ~{retry}s.")
    # Log with PII stripped so personal data never lands in logs/metrics.
    logger.info("chat persona=%s msg=%r", session.persona_id, redact_pii(req.message)[:160])

    async def factory(on_event):
        out = await session.orchestrator.handle(req.message, on_event=on_event)
        app.state.sessions.persist_memory(session)  # durable conversation memory
        return out

    return EventSourceResponse(_stream(factory))


@app.post("/pending")
async def pending(req: PendingRequest):
    session = _require_session(req.session)
    decision = {"action": req.action, "edit": req.edit or ""}

    async def factory(on_event):
        return await session.orchestrator.resume_pending(decision, on_event=on_event)

    return EventSourceResponse(_stream(factory))


@app.post("/session/docs")
async def set_docs(req: DocsRequest):
    """Replace this session's knowledge base with the demo corpus PLUS the user's own
    resume and target job description, so the Knowledge agent analyses their real data."""
    session = _require_session(req.session)
    extra: list[dict] = []
    if req.resume.strip():
        extra.append({"source": "my-resume", "content": req.resume.strip(), "access_scope": "general"})
    if req.job_description.strip():
        extra.append({"source": "jd-target", "content": req.job_description.strip(), "access_scope": "general"})
    if not extra:
        return {"ok": True, "using_custom": False}

    # Replace the demo resume/JD with the user's own; keep the generic career guides.
    base = [d for d in KNOWLEDGE_DOCS if d["source"] != "my-resume" and not d["source"].startswith("jd")]
    resources = app.state.resources
    store = await seed_knowledge_store(resources.embeddings, base + extra)
    session.orchestrator = build_session_orchestrator(
        resources, persona_id=session.persona_id, vector_store=store, sql_client=session.sql_client
    )
    return {"ok": True, "using_custom": True, "docs": [d["source"] for d in extra]}


@app.post("/session/applications")
async def set_applications(req: ApplicationsRequest):
    """Replace this session's application tracker with the user's REAL applications.

    The rows are written into the session's live SQLite database; the Analytics agent
    then answers by running genuine read-only SQL over them — no hard-coded numbers.
    """
    session = _require_session(req.session)
    apps = [a.model_dump() for a in req.applications]
    count = session.sql_client.replace_applications(apps)
    app.state.sessions.persist_applications(session)  # survive restart
    return {"ok": True, "count": count}


@app.get("/session/applications")
async def get_applications(x_session: str = Header(default="")):
    """Return the current rows so the UI can show/edit the tracker."""
    session = _require_session(x_session)
    rows = await session.sql_client.run_query(
        "SELECT company, role, status, applied_on, last_update, source, location FROM applications"
    )
    return {"applications": rows}


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
