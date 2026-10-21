"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createSession, setDocs, streamPost, warmBackend } from "@/lib/api";
import { buildTrace } from "@/lib/trace";
import type { ChatResult, Message, Pending, Persona, TraceEvent } from "@/lib/types";
import TracePanel from "@/components/TracePanel";
import PendingActionCard from "@/components/PendingActionCard";
import StarterPrompts from "@/components/StarterPrompts";
import JobResults, { JobCard } from "@/components/JobResults";
import DocsModal from "@/components/DocsModal";
import AppsModal from "@/components/AppsModal";

type Conversation = { id: string; token: string; title: string; messages: Message[] };

const CONVOS_KEY = "cp_convos";
const ACTIVE_KEY = "cp_active";

export default function ChatPage() {
  const [persona, setPersona] = useState<Persona | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [working, setWorking] = useState(true);
  const [showCitations, setShowCitations] = useState(false);
  const [showDocs, setShowDocs] = useState(false);
  const [showApps, setShowApps] = useState(false);
  const [navOpen, setNavOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const active = conversations.find((c) => c.id === activeId) || null;
  const trace = useMemo(() => buildTrace(events), [events]);

  // ---- load / persist --------------------------------------------------------
  useEffect(() => {
    warmBackend(); // wake the dyno while the user reads, so the first message isn't slow
    const p = localStorage.getItem("cp_persona");
    const tok = localStorage.getItem("cp_session");
    if (!tok) {
      window.location.href = "/";
      return;
    }
    if (p) setPersona(JSON.parse(p));

    let convos: Conversation[] = [];
    try {
      convos = JSON.parse(localStorage.getItem(CONVOS_KEY) || "[]");
    } catch {
      convos = [];
    }
    if (convos.length === 0) {
      convos = [{ id: cryptoId(), token: tok, title: "New conversation", messages: [] }];
    }
    setConversations(convos);
    const savedActive = localStorage.getItem(ACTIVE_KEY);
    setActiveId(savedActive && convos.some((c) => c.id === savedActive) ? savedActive : convos[0].id);
  }, []);

  useEffect(() => {
    if (conversations.length) localStorage.setItem(CONVOS_KEY, JSON.stringify(conversations));
  }, [conversations]);
  useEffect(() => {
    if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
  }, [activeId]);
  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [active?.messages, pending, busy]);

  // ---- conversation management ----------------------------------------------
  async function newConversation() {
    if (!persona) return;
    try {
      const token = await createSession(persona.id);
      const convo: Conversation = { id: cryptoId(), token, title: "New conversation", messages: [] };
      setConversations((cs) => [convo, ...cs]);
      setActiveId(convo.id);
      setEvents([]);
      setPending(null);
      setNavOpen(false);
    } catch {
      /* ignore */
    }
  }

  function selectConversation(id: string) {
    setActiveId(id);
    setEvents([]);
    setPending(null);
    setNavOpen(false);
  }

  function deleteConversation(id: string) {
    setConversations((cs) => {
      const next = cs.filter((c) => c.id !== id);
      if (id === activeId) {
        setActiveId(next[0]?.id || "");
        setEvents([]);
        setPending(null);
      }
      return next;
    });
  }

  function updateActive(mutate: (c: Conversation) => Conversation) {
    setConversations((cs) => cs.map((c) => (c.id === activeId ? mutate(c) : c)));
  }

  // ---- messaging -------------------------------------------------------------
  function applyResult(result: ChatResult | null) {
    if (!result) return;
    if (result.replies.length) {
      updateActive((c) => ({
        ...c,
        messages: [...c.messages, { role: "assistant", replies: result.replies, routing_reasoning: result.routing_reasoning }],
      }));
    }
    setPending(result.pending);
  }

  async function send(text: string) {
    if (!text.trim() || busy || !active) return;
    setInput("");
    updateActive((c) => ({
      ...c,
      title: c.messages.length === 0 ? truncate(text) : c.title,
      messages: [...c.messages, { role: "user", content: text }],
    }));
    setEvents([]);
    setBusy(true);
    const token = active.token;
    const systemReply = (content: string) =>
      updateActive((c) => ({
        ...c,
        messages: [...c.messages, { role: "assistant", replies: [{ agent: "system", content, metadata: {} }], routing_reasoning: "" }],
      }));
    try {
      // Retry once on failure: the first message after idle can hit a cold backend whose
      // connection resets while waking. A silent null/empty result also gets surfaced,
      // so the user is never left staring at a sent message with no response.
      let result: ChatResult | null = null;
      let lastErr: unknown = null;
      for (let attempt = 0; attempt < 2 && !result; attempt++) {
        if (attempt > 0) {
          setEvents((p) => [...p, { type: "info", message: "Backend waking up — retrying…" }]);
          await new Promise((r) => setTimeout(r, 2000));
        }
        try {
          result = await streamPost("/chat", { session: token, message: text }, (e) => setEvents((p) => [...p, e]));
        } catch (e) {
          lastErr = e;
        }
      }
      if (result && (result.replies.length || result.pending)) {
        applyResult(result);
      } else if (lastErr) {
        throw lastErr;
      } else {
        systemReply("No response came back. The backend may have been waking up — please resend.");
      }
    } catch {
      setEvents((p) => [...p, { type: "error", message: "Couldn't reach the backend — try again in ~30s." }]);
      systemReply("Couldn't reach the backend. Render's free tier sleeps when idle — wait ~30s and resend.");
    } finally {
      setBusy(false);
    }
  }

  async function resolve(action: "confirm" | "cancel" | "revise", edit?: string) {
    if (!active) return;
    setBusy(true);
    try {
      const result = await streamPost("/pending", { session: active.token, action, edit }, (e) => setEvents((p) => [...p, e]));
      applyResult(result);
    } finally {
      setBusy(false);
    }
  }

  function switchProfile() {
    localStorage.removeItem("cp_session");
    localStorage.removeItem("cp_persona");
    localStorage.removeItem(CONVOS_KEY);
    localStorage.removeItem(ACTIVE_KEY);
    window.location.href = "/";
  }

  const empty = !active || active.messages.length === 0;

  return (
    <div className="chat-page">
      {navOpen && <div className="nav-backdrop" onClick={() => setNavOpen(false)} />}
      <aside className={`conversation-sidebar ${navOpen ? "open" : ""}`}>
        <div className="conversation-sidebar-header">
          <span>Conversations</span>
          <button onClick={newConversation} aria-label="New conversation">+</button>
        </div>
        <div className="conversation-list">
          {conversations.map((c) => (
            <div key={c.id} className={`conversation-row ${c.id === activeId ? "active" : ""}`}>
              <button className="conversation-open" onClick={() => selectConversation(c.id)} title={c.title}>
                {c.title}
              </button>
              <button className="conversation-delete" onClick={() => deleteConversation(c.id)} aria-label="Delete conversation">
                ×
              </button>
            </div>
          ))}
        </div>
        <div className="sidebar-actions">
          <button className="text-button" onClick={() => { setShowDocs(true); setNavOpen(false); }}>Use my resume</button>
          <button className="text-button" onClick={() => { setShowApps(true); setNavOpen(false); }}>My applications</button>
          <a href="/outbox">Demo Outbox</a>
          <label className="toggle-label">
            <input type="checkbox" checked={working} onChange={(e) => setWorking(e.target.checked)} />
            Working
          </label>
          <label className="toggle-label">
            <input type="checkbox" checked={showCitations} onChange={(e) => setShowCitations(e.target.checked)} />
            Show citations
          </label>
          <button className="text-button" onClick={switchProfile}>Switch profile</button>
        </div>
        <div className="sidebar-foot">Multi-agent · LangGraph · MCP · RAG · SQL · HITL</div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="brand">
            <button className="nav-toggle" onClick={() => setNavOpen(true)} aria-label="Open conversations">☰</button>
            <span className="brand-logo">C</span>
            <div>
              <div className="brand-name">CareerPilot</div>
              <div className="brand-persona">
                {persona ? persona.name : "…"}
                {persona && <span className="brand-role"> · {personaTitle(persona.id)}</span>}
              </div>
            </div>
          </div>
          <div className="header-center">Job-Hunt Copilot</div>
          <div className="header-right">
            <button className="text-button" onClick={() => setShowDocs(true)}>Use my resume</button>
            <button className="text-button" onClick={() => setShowApps(true)}>My applications</button>
            <label className="toggle-label">
              <input type="checkbox" checked={working} onChange={(e) => setWorking(e.target.checked)} />
              Working
            </label>
            <label className="toggle-label">
              <input type="checkbox" checked={showCitations} onChange={(e) => setShowCitations(e.target.checked)} />
              Show citations
            </label>
            <a href="/outbox">Demo Outbox</a>
            <button className="text-button" onClick={switchProfile}>Switch profile</button>
          </div>
        </header>

        <div className="demo-banner">
          <b>Demo mode</b> — drafts are captured in the Demo Outbox, never actually sent. Inactive history expires after 24 hours.
        </div>

        <div className="chat-body" ref={scrollRef}>
          <div className="chat-scroll-inner">
            {empty && <StarterPrompts onPick={(q) => send(q)} />}

            {active?.messages.map((m, i) =>
              m.role === "user" ? (
                <div className="user-message" key={i}>
                  <span className="message-label">You</span>
                  {m.content}
                </div>
              ) : (
                m.replies.map((r, j) => (
                  <div className="agent-reply" key={`${i}-${j}`}>
                    <span className="message-label">{r.agent}</span>
                    <div className="body">{linkify(r.content)}</div>
                    {Array.isArray(r.metadata?.jobs) && (r.metadata.jobs as JobCard[]).length > 0 && active && (
                      <JobResults jobs={r.metadata.jobs as JobCard[]} token={active.token} />
                    )}
                    {showCitations && Array.isArray(r.metadata?.citations) && (r.metadata.citations as string[]).length > 0 && (
                      <div className="citations">
                        sources
                        <ul>
                          {(r.metadata.citations as string[]).map((c, k) => (
                            <li key={k}>{c}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))
              ),
            )}

            {busy && <div className="agent-reply thinking">Thinking…</div>}
            {pending && <PendingActionCard pending={pending} busy={busy} onResolve={resolve} />}
          </div>
        </div>

        <form className="chat-input-form" onSubmit={(e) => { e.preventDefault(); send(input); }}>
          <div className="chat-input-row">
            <input
              placeholder={pending ? "Resolve the pending draft above first…" : "Ask CareerPilot…"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy || !!pending}
            />
            <button type="submit" disabled={busy || !!pending || !input.trim()}>Send</button>
          </div>
        </form>
      </main>

      {working && <TracePanel trace={trace} />}

      {showApps && active && (
        <AppsModal
          token={active.token}
          onClose={() => setShowApps(false)}
          onSaved={() =>
            updateActive((c) => ({
              ...c,
              messages: [
                ...c.messages,
                { role: "assistant", replies: [{ agent: "system", content: "Saved your applications. The Analytics agent now runs real SQL over them — try: \"How many of my applications got a response, and which are stale?\"", metadata: {} }], routing_reasoning: "" },
              ],
            }))
          }
        />
      )}

      {showDocs && (
        <DocsModal
          onClose={() => setShowDocs(false)}
          onSave={async (resume, jd) => {
            if (!active) return;
            await setDocs(active.token, resume, jd);
            updateActive((c) => ({
              ...c,
              messages: [
                ...c.messages,
                { role: "assistant", replies: [{ agent: "system", content: "Got it — I'll analyse your résumé and that job description from now on. Try: \"Does my resume fit this role? What are my gaps?\"", metadata: {} }], routing_reasoning: "" },
              ],
            }));
          }}
        />
      )}
    </div>
  );
}

function personaTitle(id: string): string {
  return id === "riya" ? "Software Engineer" : "Senior Engineer · Career Switcher";
}

function linkify(text: string): (string | React.ReactNode)[] {
  return text.split(/(https?:\/\/[^\s]+)/g).map((part, i) =>
    /^https?:\/\//.test(part) ? (
      <a key={i} href={part} target="_blank" rel="noreferrer">
        {part}
      </a>
    ) : (
      part
    ),
  );
}

function truncate(s: string, n = 40): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function cryptoId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
